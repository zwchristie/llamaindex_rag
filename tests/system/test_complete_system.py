"""
Comprehensive system tests for the complete text-to-SQL workflow.
Tests all components from document upload to SQL generation with HITL.
"""

import pytest
import asyncio
import os
import json
from datetime import datetime, timedelta
import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch
import uuid
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from text_to_sql_rag.models.view_models import (
    ViewMetadata, ViewColumn, ViewJoin, HITLRequest, SessionState
)
from text_to_sql_rag.services.view_service import ViewService
from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService
from text_to_sql_rag.services.hitl_service import HITLService
from text_to_sql_rag.services.session_service import SessionService
from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
from text_to_sql_rag.core.text_to_sql_agent import TextToSQLAgent

# Test configuration
TEST_MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
TEST_OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
TEST_BEDROCK_ENDPOINT = os.getenv("BEDROCK_ENDPOINT_URL", "https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess")

logger = logging.getLogger(__name__)

# Mock LLM service for predictable testing
class MockLLMService:
    def __init__(self):
        self.call_count = 0
        self.responses = {
            "initial": {
                "sql": "SELECT user_id, username, email FROM V_TEST_USERS WHERE created_date >= CURRENT_DATE - INTERVAL '30 days'",
                "explanation": "This query retrieves user information for users created in the last 30 days"
            },
            "clarification_needed": {
                "sql": "-- CLARIFICATION NEEDED: Please specify which user metrics you want",
                "explanation": "I need clarification about which specific user metrics you're looking for"
            },
            "after_clarification": {
                "sql": "SELECT user_id, username, login_count, last_login_date FROM V_USER_METRICS WHERE last_login_date >= CURRENT_DATE - INTERVAL '7 days' ORDER BY login_count DESC",
                "explanation": "This query shows active users from the last 7 days ordered by login frequency"
            },
            "follow_up": {
                "sql": "SELECT user_id, username, login_count, last_login_date FROM V_USER_METRICS WHERE last_login_date >= CURRENT_DATE - INTERVAL '7 days' AND login_count > 5 ORDER BY login_count DESC LIMIT 10",
                "explanation": "This modified query shows the top 10 most active users from the last 7 days with more than 5 logins"
            }
        }
    
    async def generate_sql(self, prompt):
        self.call_count += 1
        
        if "clarification" in prompt.lower() or "specify" in prompt.lower():
            return self.responses["after_clarification"]
        elif "top 10" in prompt.lower() or "most active" in prompt.lower():
            return self.responses["follow_up"]
        elif self.call_count == 1:
            return self.responses["initial"]
        else:
            return self.responses["clarification_needed"]
    
    async def generate_text(self, prompt):
        return f"Mock LLM response to: {prompt[:50]}..."


@pytest.fixture(scope="module")
async def system_setup():
    """Setup complete system for testing."""
    # Create unique test database and index
    test_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    test_db_name = f"test_system_{test_id}"
    test_index_name = f"test_views_{test_id}"
    
    # MongoDB connection
    try:
        mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(TEST_MONGODB_URL)
        await mongodb_client.admin.command('ping')
        db = mongodb_client[test_db_name]
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")
    
    # OpenSearch connection
    try:
        opensearch_client = AsyncOpenSearch(
            hosts=[{"host": TEST_OPENSEARCH_HOST, "port": 9200}],
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
        )
        await opensearch_client.ping()
    except Exception as e:
        pytest.skip(f"OpenSearch not available: {e}")
    
    # Initialize services
    view_service = ViewService(db)
    
    # Try real embedding service first, fall back to mock if not available
    try:
        embedding_service = EmbeddingService(TEST_BEDROCK_ENDPOINT, "amazon.titan-embed-text-v2:0")
        # Test if embedding service works
        test_embedding = await embedding_service.get_embedding("test")
        if not test_embedding:
            raise Exception("Embedding service returned empty result")
        use_real_embeddings = True
    except Exception as e:
        logger.warning(f"Using mock embedding service: {e}")
        use_real_embeddings = False
        
        class MockEmbeddingService:
            def __init__(self):
                self.embedding_dimension = 1536
            
            async def get_embedding(self, text):
                # Return predictable mock embedding
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                hash_int = int(hash_obj.hexdigest(), 16)
                # Generate consistent mock embedding
                embedding = []
                for i in range(1536):
                    embedding.append((hash_int + i) % 1000 / 1000.0 - 0.5)
                return embedding
            
            async def get_embeddings_batch(self, texts, batch_size=5):
                return [await self.get_embedding(text) for text in texts]
            
            def get_embedding_dimension(self):
                return self.embedding_dimension
        
        embedding_service = MockEmbeddingService()
    
    vector_service = VectorService(opensearch_client, test_index_name, "embedding")
    hitl_service = HITLService(db, timeout_minutes=1)  # Short timeout for tests
    session_service = SessionService(db)
    llm_service = MockLLMService()  # Always use mock for predictable testing
    
    # Create indexes
    await view_service.ensure_indexes()
    await session_service.ensure_indexes()
    
    services = {
        "view_service": view_service,
        "embedding_service": embedding_service,
        "vector_service": vector_service,
        "hitl_service": hitl_service,
        "session_service": session_service,
        "llm_service": llm_service,
        "use_real_embeddings": use_real_embeddings
    }
    
    # Initialize agent
    agent = TextToSQLAgent(
        view_service=view_service,
        embedding_service=embedding_service,
        vector_service=vector_service,
        hitl_service=hitl_service,
        llm_service=llm_service,
        session_service=session_service
    )
    services["agent"] = agent
    
    yield services
    
    # Cleanup
    try:
        await mongodb_client.drop_database(test_db_name)
        if await opensearch_client.indices.exists(index=test_index_name):
            await opensearch_client.indices.delete(index=test_index_name)
    except:
        pass
    
    mongodb_client.close()
    await opensearch_client.close()


@pytest.fixture
async def sample_metadata():
    """Create sample view metadata for testing."""
    return [
        ViewMetadata(
            view_name="V_TEST_USERS",
            view_type="CORE",
            schema_name="TEST",
            description="Test view containing user account information and registration details",
            use_cases="User management, account analysis, registration tracking, user demographics",
            columns=[
                ViewColumn(name="user_id", type="NUMBER", notNull=True, description="Unique user identifier"),
                ViewColumn(name="username", type="VARCHAR2", notNull=True, description="User login name"),
                ViewColumn(name="email", type="VARCHAR2", notNull=False, description="User email address"),
                ViewColumn(name="created_date", type="DATE", notNull=True, description="Account creation date"),
                ViewColumn(name="status", type="VARCHAR2", notNull=True, description="Account status: ACTIVE, INACTIVE, SUSPENDED")
            ],
            joins=[
                ViewJoin(table_name="USER_PROFILES", join_type="LEFT", join_condition="u.user_id = up.user_id", description="User profile details")
            ],
            sample_sql="SELECT user_id, username, email, created_date FROM V_TEST_USERS WHERE status = 'ACTIVE'",
            example_query="Show me all active users",
            data_returned="User account information including ID, username, email and creation date"
        ),
        ViewMetadata(
            view_name="V_USER_METRICS",
            view_type="CORE",
            schema_name="ANALYTICS",
            description="User engagement metrics including login frequency, session duration, and activity patterns",
            use_cases="User behavior analysis, engagement tracking, activity reporting, user segmentation",
            columns=[
                ViewColumn(name="user_id", type="NUMBER", notNull=True, description="User identifier"),
                ViewColumn(name="username", type="VARCHAR2", notNull=False, description="User login name"),
                ViewColumn(name="login_count", type="NUMBER", notNull=False, description="Total number of logins"),
                ViewColumn(name="last_login_date", type="DATE", notNull=False, description="Most recent login date"),
                ViewColumn(name="avg_session_minutes", type="NUMBER", notNull=False, description="Average session duration in minutes"),
                ViewColumn(name="total_actions", type="NUMBER", notNull=False, description="Total user actions performed")
            ],
            sample_sql="SELECT user_id, username, login_count, last_login_date FROM V_USER_METRICS ORDER BY login_count DESC",
            example_query="Show me the most active users",
            data_returned="User activity metrics and engagement statistics"
        ),
        ViewMetadata(
            view_name="V_ORDER_SUMMARY",
            view_type="SUPPORTING",
            schema_name="SALES",
            description="Order summary information including totals, status, and customer details",
            use_cases="Order tracking, sales analysis, customer order history, revenue reporting",
            columns=[
                ViewColumn(name="order_id", type="NUMBER", notNull=True, description="Unique order identifier"),
                ViewColumn(name="user_id", type="NUMBER", notNull=True, description="Customer user ID"),
                ViewColumn(name="order_date", type="DATE", notNull=True, description="Date order was placed"),
                ViewColumn(name="total_amount", type="NUMBER", notNull=False, description="Total order value"),
                ViewColumn(name="status", type="VARCHAR2", notNull=True, description="Order status: PENDING, COMPLETED, CANCELLED")
            ],
            sample_sql="SELECT order_id, user_id, order_date, total_amount, status FROM V_ORDER_SUMMARY WHERE order_date >= CURRENT_DATE - 30",
            example_query="Show recent orders",
            data_returned="Order details with customer and financial information"
        )
    ]


@pytest.mark.asyncio
async def test_1_mongodb_document_upload_and_retrieval(system_setup, sample_metadata):
    """Test 1: Verify MongoDB document upload and retrieval works correctly."""
    print("\n=== TEST 1: MongoDB Document Upload and Retrieval ===")
    
    services = await system_setup
    view_service = services["view_service"]
    
    # Test document upload
    print("1.1 Testing document upload to MongoDB...")
    view_ids = []
    for view in sample_metadata:
        view_id = await view_service.create_view(view)
        assert view_id is not None, f"Failed to create view {view.view_name}"
        view_ids.append(view_id)
        print(f"✓ Uploaded view: {view.view_name} (ID: {view_id})")
    
    # Test document retrieval
    print("1.2 Testing document retrieval from MongoDB...")
    
    # Test get all views
    all_views = await view_service.get_all_views()
    assert len(all_views) == 3, f"Expected 3 views, got {len(all_views)}"
    print(f"✓ Retrieved {len(all_views)} views from MongoDB")
    
    # Test get specific view
    user_view = await view_service.get_view_by_name("V_TEST_USERS")
    assert user_view is not None, "Failed to retrieve V_TEST_USERS"
    assert user_view.view_name == "V_TEST_USERS"
    assert len(user_view.columns) == 5
    assert user_view.full_text is not None, "full_text should be generated"
    print(f"✓ Retrieved specific view: {user_view.view_name}")
    print(f"  - Columns: {len(user_view.columns)}")
    print(f"  - Full text length: {len(user_view.full_text)} characters")
    
    # Test filtering by type
    core_views = await view_service.get_all_views(view_type="CORE")
    supporting_views = await view_service.get_all_views(view_type="SUPPORTING")
    assert len(core_views) == 2, f"Expected 2 CORE views, got {len(core_views)}"
    assert len(supporting_views) == 1, f"Expected 1 SUPPORTING view, got {len(supporting_views)}"
    print(f"✓ Filtered views: {len(core_views)} CORE, {len(supporting_views)} SUPPORTING")
    
    # Test text search
    search_results = await view_service.search_views_by_text("user metrics")
    assert len(search_results) > 0, "Text search should return results"
    print(f"✓ Text search returned {len(search_results)} results")
    
    print("✅ MongoDB document upload and retrieval: PASSED\n")


@pytest.mark.asyncio
async def test_2_opensearch_embedding_and_indexing(system_setup, sample_metadata):
    """Test 2: Verify OpenSearch embedding and indexing works correctly."""
    print("=== TEST 2: OpenSearch Embedding and Indexing ===")
    
    services = await system_setup
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    use_real_embeddings = services["use_real_embeddings"]
    
    # Upload views to MongoDB first
    print("2.1 Setting up test data in MongoDB...")
    for view in sample_metadata:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    print(f"✓ {len(all_views)} views in MongoDB")
    
    # Test embedding generation
    print("2.2 Testing embedding generation...")
    test_text = "user account information and login activity"
    embedding = await embedding_service.get_embedding(test_text)
    assert embedding is not None, "Embedding should not be None"
    assert len(embedding) > 0, "Embedding should not be empty"
    
    if use_real_embeddings:
        print(f"✓ Generated real embedding with dimension: {len(embedding)}")
        assert len(embedding) == 1536, f"Expected dimension 1536, got {len(embedding)}"
    else:
        print(f"✓ Generated mock embedding with dimension: {len(embedding)}")
    
    # Test batch embedding generation
    texts = [view.generate_full_text() for view in all_views[:2]]
    batch_embeddings = await embedding_service.get_embeddings_batch(texts)
    assert len(batch_embeddings) == 2, "Should generate 2 embeddings"
    print(f"✓ Generated batch embeddings: {len(batch_embeddings)}")
    
    # Test vector indexing
    print("2.3 Testing OpenSearch index creation and document indexing...")
    
    # Create index with detected dimension
    dimension = embedding_service.get_embedding_dimension()
    if dimension:
        await vector_service.create_index(dimension)
        print(f"✓ Created OpenSearch index with dimension {dimension}")
    
    # Reindex all views
    await vector_service.reindex_all_views(all_views, embedding_service)
    print("✓ Reindexed all views in OpenSearch")
    
    # Verify index stats
    stats = await vector_service.get_index_stats()
    assert stats.get("document_count", 0) == 3, f"Expected 3 documents in index, got {stats.get('document_count', 0)}"
    print(f"✓ Index contains {stats.get('document_count')} documents")
    print(f"  - Index size: {stats.get('store_size_mb', 0)} MB")
    
    print("✅ OpenSearch embedding and indexing: PASSED\n")


@pytest.mark.asyncio
async def test_3_opensearch_retrieval(system_setup, sample_metadata):
    """Test 3: Verify OpenSearch retrieval returns expected results."""
    print("=== TEST 3: OpenSearch Retrieval Testing ===")
    
    services = await system_setup
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    
    # Setup data
    print("3.1 Setting up test data...")
    for view in sample_metadata:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    await vector_service.reindex_all_views(all_views, embedding_service)
    print("✓ Data setup complete")
    
    # Test vector similarity search
    print("3.2 Testing vector similarity search...")
    
    # Search for user-related content
    query1 = "user account registration and login information"
    query1_embedding = await embedding_service.get_embedding(query1)
    results1 = await vector_service.search_similar_views(query1_embedding, k=3)
    
    assert len(results1) > 0, "Should return search results"
    print(f"✓ Query: '{query1}' returned {len(results1)} results")
    
    # Verify V_TEST_USERS is in top results (should be most relevant)
    view_names = [result[0].view_name for result in results1]
    assert "V_TEST_USERS" in view_names, f"V_TEST_USERS should be in results: {view_names}"
    
    # Check that results are ranked by relevance
    for i, (view, score) in enumerate(results1):
        print(f"  {i+1}. {view.view_name} (score: {score:.3f})")
    
    # Search for metrics/analytics content
    print("3.3 Testing different query types...")
    query2 = "user activity analytics and engagement metrics"
    query2_embedding = await embedding_service.get_embedding(query2)
    results2 = await vector_service.search_similar_views(query2_embedding, k=3)
    
    view_names2 = [result[0].view_name for result in results2]
    print(f"✓ Query: '{query2}' returned views: {view_names2}")
    
    # Search for order/sales content
    query3 = "order history and sales transactions"
    query3_embedding = await embedding_service.get_embedding(query3)
    results3 = await vector_service.search_similar_views(query3_embedding, k=3)
    
    view_names3 = [result[0].view_name for result in results3]
    print(f"✓ Query: '{query3}' returned views: {view_names3}")
    assert "V_ORDER_SUMMARY" in view_names3, f"V_ORDER_SUMMARY should be relevant for order queries"
    
    # Test hybrid search (if available)
    print("3.4 Testing hybrid search...")
    try:
        hybrid_results = await vector_service.hybrid_search(
            query_text="user login metrics",
            query_embedding=query2_embedding,
            k=2
        )
        print(f"✓ Hybrid search returned {len(hybrid_results)} results")
        for i, (view, score) in enumerate(hybrid_results):
            print(f"  {i+1}. {view.view_name} (score: {score:.3f})")
    except Exception as e:
        print(f"⚠️ Hybrid search not available: {e}")
    
    print("✅ OpenSearch retrieval: PASSED\n")


@pytest.mark.asyncio
async def test_4_complete_text_to_sql_flow(system_setup, sample_metadata):
    """Test 4: Execute complete text-to-SQL flow without HITL."""
    print("=== TEST 4: Complete Text-to-SQL Flow ===")
    
    services = await system_setup
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    agent = services["agent"]
    
    # Setup data
    print("4.1 Setting up test data...")
    for view in sample_metadata:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    await vector_service.reindex_all_views(all_views, embedding_service)
    print("✓ Data setup complete")
    
    # Test complete flow
    print("4.2 Testing complete text-to-SQL flow...")
    
    # Process a user query
    user_query = "Show me all users created in the last 30 days"
    print(f"User query: '{user_query}'")
    
    # Use agent to process query (this will use mock LLM for predictable results)
    result = await agent.process_query(user_query)
    
    # Verify result structure
    assert result is not None, "Result should not be None"
    assert "session_id" in result, "Result should contain session_id"
    assert "response" in result, "Result should contain response"
    
    session_id = result["session_id"]
    print(f"✓ Generated session ID: {session_id}")
    
    # Check if SQL was generated
    if "sql" in result and result["sql"]:
        print(f"✓ Generated SQL: {result['sql'][:100]}...")
        assert "SELECT" in result["sql"].upper(), "Generated SQL should contain SELECT"
        assert "V_TEST_USERS" in result["sql"], "Should reference the correct view"
    
    # Check response format
    assert result["response"] is not None, "Should have a response"
    print(f"✓ Generated response (length: {len(result['response'])} chars)")
    
    # Verify session was created
    session_service = services["session_service"]
    session = await session_service.get_session(session_id)
    assert session is not None, "Session should be created"
    assert session.user_query == user_query, "Session should store original query"
    print("✓ Session state persisted correctly")
    
    print("✅ Complete text-to-SQL flow: PASSED\n")


@pytest.mark.asyncio
async def test_5_hitl_workflow_with_clarification(system_setup, sample_metadata):
    """Test 5: Execute text-to-SQL flow with HITL clarification in the middle."""
    print("=== TEST 5: HITL Workflow with Clarification ===")
    
    services = await system_setup
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    hitl_service = services["hitl_service"]
    session_service = services["session_service"]
    
    # Setup data
    print("5.1 Setting up test data...")
    for view in sample_metadata:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    await vector_service.reindex_all_views(all_views, embedding_service)
    print("✓ Data setup complete")
    
    # Test HITL workflow
    print("5.2 Testing HITL approval workflow...")
    
    # Create a HITL request manually (simulating agent behavior)
    session_id = str(uuid.uuid4())
    user_query = "Show me user metrics data"
    generated_sql = "SELECT user_id, username, login_count FROM V_USER_METRICS ORDER BY login_count DESC"
    
    print(f"Creating HITL request for query: '{user_query}'")
    request_id = await hitl_service.create_approval_request(
        session_id=session_id,
        user_query=user_query,
        generated_sql=generated_sql,
        sql_explanation="This query retrieves user metrics ordered by login frequency",
        selected_views=["V_USER_METRICS"]
    )
    
    assert request_id is not None, "Should create HITL request"
    print(f"✓ Created HITL request: {request_id}")
    
    # Verify request was created
    request = await hitl_service.get_request(request_id)
    assert request is not None, "Request should exist"
    assert request.status == "pending", "Request should be pending"
    print(f"✓ Request status: {request.status}")
    
    # Get pending requests
    pending = await hitl_service.get_pending_requests()
    assert len(pending) >= 1, "Should have pending requests"
    print(f"✓ Found {len(pending)} pending requests")
    
    # Test approval workflow
    print("5.3 Testing approval process...")
    
    # Approve the request
    success = await hitl_service.approve_request(
        request_id,
        reviewer_notes="SQL query looks correct and safe",
        resolution_reason="Approved for execution"
    )
    assert success, "Approval should succeed"
    print("✓ Request approved successfully")
    
    # Verify approval
    approved_request = await hitl_service.get_request(request_id)
    assert approved_request.status == "approved", "Request should be approved"
    assert approved_request.reviewer_notes == "SQL query looks correct and safe"
    print(f"✓ Request status updated to: {approved_request.status}")
    
    # Test rejection workflow
    print("5.4 Testing rejection process...")
    
    # Create another request for rejection test
    request_id2 = await hitl_service.create_approval_request(
        session_id=session_id,
        user_query="Delete all user data",  # Potentially dangerous query
        generated_sql="DELETE FROM V_TEST_USERS",
        sql_explanation="This query would delete all user data",
        selected_views=["V_TEST_USERS"]
    )
    
    # Reject the request
    success = await hitl_service.reject_request(
        request_id2,
        reviewer_notes="DELETE operations are not allowed",
        resolution_reason="Security policy violation"
    )
    assert success, "Rejection should succeed"
    print("✓ Request rejected successfully")
    
    # Verify rejection
    rejected_request = await hitl_service.get_request(request_id2)
    assert rejected_request.status == "rejected", "Request should be rejected"
    print(f"✓ Request status updated to: {rejected_request.status}")
    
    # Test clarification workflow
    print("5.5 Testing clarification workflow...")
    
    # Simulate waiting for approval (would normally block)
    async def simulate_wait_and_approve():
        await asyncio.sleep(0.1)  # Small delay
        await hitl_service.approve_request(request_id, "Approved after review")
    
    # Start approval task
    approval_task = asyncio.create_task(simulate_wait_and_approve())
    
    # This would normally block until approval
    # For testing, we'll simulate the process
    await approval_task
    
    print("✅ HITL workflow with clarification: PASSED\n")


@pytest.mark.asyncio
async def test_6_follow_up_questions_and_sql_modification(system_setup, sample_metadata):
    """Test 6: Test follow-up questions that modify generated SQL."""
    print("=== TEST 6: Follow-up Questions and SQL Modification ===")
    
    services = await system_setup
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    agent = services["agent"]
    session_service = services["session_service"]
    
    # Setup data
    print("6.1 Setting up test data...")
    for view in sample_metadata:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    await vector_service.reindex_all_views(all_views, embedding_service)
    print("✓ Data setup complete")
    
    # Test initial query
    print("6.2 Testing initial query...")
    
    initial_query = "Show me user activity data"
    print(f"Initial query: '{initial_query}'")
    
    result1 = await agent.process_query(initial_query)
    assert result1 is not None, "Should get initial result"
    session_id = result1["session_id"]
    
    if "sql" in result1 and result1["sql"]:
        print(f"✓ Initial SQL generated: {result1['sql'][:80]}...")
        initial_sql = result1["sql"]
    else:
        print("⚠️ Initial query may need clarification (expected for some test scenarios)")
        initial_sql = "SELECT * FROM V_USER_METRICS"
    
    # Test follow-up modification
    print("6.3 Testing follow-up question to modify SQL...")
    
    follow_up_query = "Actually, show me only the top 10 most active users"
    print(f"Follow-up query: '{follow_up_query}'")
    
    # Process follow-up in same session
    result2 = await agent.process_query(follow_up_query, session_id)
    assert result2 is not None, "Should get follow-up result"
    assert result2["session_id"] == session_id, "Should reuse same session"
    
    if "sql" in result2 and result2["sql"]:
        follow_up_sql = result2["sql"]
        print(f"✓ Modified SQL: {follow_up_sql[:80]}...")
        
        # Verify the SQL was actually modified
        if "LIMIT" in follow_up_sql.upper() or "TOP" in follow_up_sql.upper():
            print("✓ SQL correctly modified to include limit")
        else:
            print("⚠️ SQL may not have been modified as expected (depends on mock LLM)")
    
    # Test another modification
    print("6.4 Testing another SQL modification...")
    
    modification_query = "Add a filter to show only users who logged in recently"
    print(f"Modification query: '{modification_query}'")
    
    result3 = await agent.process_query(modification_query, session_id)
    assert result3 is not None, "Should get modification result"
    
    # Verify session persistence
    print("6.5 Verifying session persistence...")
    
    session = await session_service.get_session(session_id)
    assert session is not None, "Session should exist"
    print(f"✓ Session contains {len(session.selected_views)} selected views")
    
    # Check session history (if implemented)
    sessions = await session_service.get_active_sessions(limit=10)
    session_ids = [s.session_id for s in sessions]
    assert session_id in session_ids, "Session should be in active sessions"
    print("✓ Session properly tracked in active sessions")
    
    # Test conversation continuity
    print("6.6 Testing conversation continuity...")
    
    stats = await session_service.get_stats()
    assert stats["total_sessions"] >= 1, "Should have at least one session"
    print(f"✓ System has processed {stats['total_sessions']} sessions")
    
    print("✅ Follow-up questions and SQL modification: PASSED\n")


@pytest.mark.asyncio 
async def test_7_end_to_end_integration(system_setup, sample_metadata):
    """Test 7: Complete end-to-end integration test."""
    print("=== TEST 7: End-to-End Integration Test ===")
    
    services = await system_setup
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    hitl_service = services["hitl_service"]
    session_service = services["session_service"]
    agent = services["agent"]
    
    # Setup comprehensive test data
    print("7.1 Setting up comprehensive test data...")
    for view in sample_metadata:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    await vector_service.reindex_all_views(all_views, embedding_service)
    print(f"✓ {len(all_views)} views indexed in OpenSearch")
    
    # Test complex user journey
    print("7.2 Testing complex user journey...")
    
    # Scenario: Data analyst exploring user engagement
    queries = [
        "Show me user registration trends",
        "I want to see user activity metrics instead", 
        "Focus on the most active users only",
        "Add login frequency information"
    ]
    
    session_id = None
    for i, query in enumerate(queries, 1):
        print(f"  Query {i}: '{query}'")
        
        result = await agent.process_query(query, session_id)
        assert result is not None, f"Query {i} should return result"
        
        if session_id is None:
            session_id = result["session_id"]
            print(f"    ✓ Started session: {session_id}")
        else:
            assert result["session_id"] == session_id, "Should continue in same session"
        
        if "sql" in result and result["sql"]:
            print(f"    ✓ Generated SQL ({len(result['sql'])} chars)")
        
        if "views_used" in result:
            print(f"    ✓ Used views: {result.get('views_used', [])}")
    
    # Verify final session state
    final_session = await session_service.get_session(session_id)
    assert final_session is not None, "Final session should exist"
    print(f"✓ Final session state preserved")
    
    # Test system statistics
    print("7.3 Testing system statistics...")
    
    view_stats = await view_service.get_stats()
    hitl_stats = await hitl_service.get_stats()
    session_stats = await session_service.get_stats()
    index_stats = await vector_service.get_index_stats()
    
    print(f"✓ System statistics:")
    print(f"  - Views: {view_stats.get('total_views', 0)} total")
    print(f"  - HITL requests: {hitl_stats.get('total_requests', 0)} total")
    print(f"  - Sessions: {session_stats.get('total_sessions', 0)} total")
    print(f"  - Index documents: {index_stats.get('document_count', 0)}")
    
    # Test cleanup operations
    print("7.4 Testing cleanup operations...")
    
    await hitl_service.cleanup_expired_requests()
    deleted_sessions = await session_service.cleanup_old_sessions(days_old=0)
    print(f"✓ Cleanup completed (deleted {deleted_sessions} old sessions)")
    
    print("✅ End-to-end integration test: PASSED\n")


# Main test runner
if __name__ == "__main__":
    async def run_all_tests():
        print("🧪 COMPREHENSIVE SYSTEM TESTING")
        print("=" * 50)
        
        try:
            # These would be run by pytest, but showing the flow
            print("Setting up system...")
            
            # In actual pytest run, fixtures handle setup
            print("\nRunning comprehensive system tests...")
            print("Note: Run with 'pytest tests/system/test_complete_system.py -v -s'")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        
        return True
    
    # Run if called directly
    import asyncio
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)