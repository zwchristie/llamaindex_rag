"""
Integration tests for the complete text-to-SQL flow.
"""

import pytest
import asyncio
import os
from datetime import datetime
import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from text_to_sql_rag.models.view_models import ViewMetadata, ViewColumn, HITLRequest
from text_to_sql_rag.services.view_service import ViewService
from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService
from text_to_sql_rag.services.hitl_service import HITLService
from text_to_sql_rag.services.session_service import SessionService
from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
from text_to_sql_rag.core.text_to_sql_agent import TextToSQLAgent


# Test configuration
TEST_MONGODB_URL = os.getenv("TEST_MONGODB_URL", "mongodb://admin:password@localhost:27017")
TEST_OPENSEARCH_HOST = os.getenv("TEST_OPENSEARCH_HOST", "localhost")
TEST_BEDROCK_ENDPOINT = os.getenv("BEDROCK_ENDPOINT_URL", "https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess")

# Skip integration tests if services are not available
pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
async def mongodb_client():
    """Create MongoDB client for testing."""
    client = motor.motor_asyncio.AsyncIOMotorClient(TEST_MONGODB_URL)
    
    # Test connection
    try:
        await client.admin.command('ping')
        yield client
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")
    finally:
        client.close()


@pytest.fixture(scope="session")
async def opensearch_client():
    """Create OpenSearch client for testing."""
    client = AsyncOpenSearch(
        hosts=[{"host": TEST_OPENSEARCH_HOST, "port": 9200}],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
    )
    
    # Test connection
    try:
        await client.ping()
        yield client
    except Exception as e:
        pytest.skip(f"OpenSearch not available: {e}")
    finally:
        await client.close()


@pytest.fixture
async def test_database(mongodb_client):
    """Create test database and cleanup after test."""
    db_name = f"test_text_to_sql_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    db = mongodb_client[db_name]
    
    yield db
    
    # Cleanup
    await mongodb_client.drop_database(db_name)


@pytest.fixture
async def test_index_name():
    """Generate unique test index name."""
    return f"test_views_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


@pytest.fixture
async def services(test_database, opensearch_client, test_index_name):
    """Initialize all services for testing."""
    try:
        # Initialize services
        view_service = ViewService(test_database)
        embedding_service = EmbeddingService(
            TEST_BEDROCK_ENDPOINT,
            "amazon.titan-embed-text-v2:0"
        )
        vector_service = VectorService(opensearch_client, test_index_name, "embedding")
        hitl_service = HITLService(test_database, timeout_minutes=1)  # Short timeout for tests
        session_service = SessionService(test_database)
        llm_service = BedrockEndpointService(
            TEST_BEDROCK_ENDPOINT,
            "anthropic.claude-3-haiku-20240307-v1:0"
        )
        
        # Create indexes
        await view_service.ensure_indexes()
        await session_service.ensure_indexes()
        
        services = {
            "view_service": view_service,
            "embedding_service": embedding_service,
            "vector_service": vector_service,
            "hitl_service": hitl_service,
            "session_service": session_service,
            "llm_service": llm_service
        }
        
        yield services
        
    except Exception as e:
        pytest.skip(f"Failed to initialize services: {e}")
    finally:
        # Cleanup OpenSearch index
        try:
            if await opensearch_client.indices.exists(index=test_index_name):
                await opensearch_client.indices.delete(index=test_index_name)
        except:
            pass


@pytest.fixture
async def sample_views():
    """Create sample views for testing."""
    return [
        ViewMetadata(
            view_name="V_TEST_USERS",
            view_type="CORE",
            description="Test view for user data",
            use_cases="User queries and analysis",
            columns=[
                ViewColumn(name="user_id", type="NUMBER", notNull=True, description="Unique user ID"),
                ViewColumn(name="username", type="VARCHAR2", notNull=True, description="User login name"),
                ViewColumn(name="email", type="VARCHAR2", notNull=False, description="User email address"),
                ViewColumn(name="created_date", type="DATE", notNull=True, description="Account creation date")
            ]
        ),
        ViewMetadata(
            view_name="V_TEST_ORDERS",
            view_type="CORE", 
            description="Test view for order data",
            use_cases="Order analysis and reporting",
            columns=[
                ViewColumn(name="order_id", type="NUMBER", notNull=True, description="Unique order ID"),
                ViewColumn(name="user_id", type="NUMBER", notNull=True, description="User who placed order"),
                ViewColumn(name="order_date", type="DATE", notNull=True, description="Date order was placed"),
                ViewColumn(name="total_amount", type="NUMBER", notNull=False, description="Total order amount")
            ]
        )
    ]


@pytest.mark.asyncio
async def test_view_creation_and_retrieval(services, sample_views):
    """Test creating and retrieving views."""
    view_service = services["view_service"]
    
    # Create views
    for view in sample_views:
        view_id = await view_service.create_view(view)
        assert view_id is not None
    
    # Retrieve all views
    all_views = await view_service.get_all_views()
    assert len(all_views) == 2
    
    # Retrieve specific view
    user_view = await view_service.get_view_by_name("V_TEST_USERS")
    assert user_view is not None
    assert user_view.view_name == "V_TEST_USERS"
    assert len(user_view.columns) == 4


@pytest.mark.asyncio
async def test_embedding_and_vector_search(services, sample_views):
    """Test embedding generation and vector search."""
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    
    # Skip if embedding service is not available
    try:
        test_embedding = await embedding_service.get_embedding("test")
        if not test_embedding:
            pytest.skip("Embedding service not available")
    except Exception as e:
        pytest.skip(f"Embedding service not available: {e}")
    
    # Create views in database
    for view in sample_views:
        await view_service.create_view(view)
    
    # Get all views and create embeddings
    all_views = await view_service.get_all_views()
    
    # Create index and reindex views
    await vector_service.reindex_all_views(all_views, embedding_service)
    
    # Test vector search
    query_embedding = await embedding_service.get_embedding("user information")
    results = await vector_service.search_similar_views(query_embedding, k=2)
    
    assert len(results) > 0
    # Should find the user view with higher relevance
    assert any(result[0].view_name == "V_TEST_USERS" for result in results)


@pytest.mark.asyncio  
async def test_hitl_workflow(services):
    """Test HITL approval workflow."""
    hitl_service = services["hitl_service"]
    
    # Create approval request
    request_id = await hitl_service.create_approval_request(
        session_id="test-session",
        user_query="Show me all users",
        generated_sql="SELECT * FROM V_TEST_USERS",
        sql_explanation="This query selects all columns from the test users view",
        selected_views=["V_TEST_USERS"]
    )
    
    assert request_id is not None
    
    # Get the request
    request = await hitl_service.get_request(request_id)
    assert request is not None
    assert request.status == "pending"
    
    # Get pending requests
    pending = await hitl_service.get_pending_requests()
    assert len(pending) == 1
    assert pending[0].request_id == request_id
    
    # Approve the request
    success = await hitl_service.approve_request(
        request_id,
        reviewer_notes="Looks good",
        resolution_reason="Approved by integration test"
    )
    assert success is True
    
    # Verify approval
    approved_request = await hitl_service.get_request(request_id)
    assert approved_request.status == "approved"
    assert approved_request.reviewer_notes == "Looks good"


@pytest.mark.asyncio
async def test_session_management(services):
    """Test session state management."""
    session_service = services["session_service"]
    
    from text_to_sql_rag.models.view_models import SessionState
    
    # Create session state
    session = SessionState(
        session_id="test-session-123",
        current_step="retrieve_views",
        user_query="Test query",
        retrieved_views=[],
        selected_views=["V_TEST_USERS"]
    )
    
    # Save session
    success = await session_service.save_session(session)
    assert success is True
    
    # Retrieve session
    retrieved_session = await session_service.get_session("test-session-123")
    assert retrieved_session is not None
    assert retrieved_session.session_id == "test-session-123"
    assert retrieved_session.user_query == "Test query"
    
    # Update session step
    success = await session_service.update_session_step(
        "test-session-123",
        "generate_sql",
        {"generated_sql": "SELECT * FROM V_TEST_USERS"}
    )
    assert success is True
    
    # Get active sessions
    active_sessions = await session_service.get_active_sessions()
    assert len(active_sessions) >= 1
    
    # Get stats
    stats = await session_service.get_stats()
    assert stats["total_sessions"] >= 1


@pytest.mark.asyncio
async def test_complete_text_to_sql_flow(services, sample_views):
    """Test the complete text-to-SQL flow with mocked LLM."""
    view_service = services["view_service"]
    embedding_service = services["embedding_service"]
    vector_service = services["vector_service"]
    hitl_service = services["hitl_service"]
    session_service = services["session_service"]
    
    # Mock LLM service for predictable results
    class MockLLMService:
        async def generate_sql(self, prompt):
            return {
                "sql": "SELECT user_id, username, email FROM V_TEST_USERS WHERE created_date >= SYSDATE - 30",
                "explanation": "This query retrieves user information for users created in the last 30 days"
            }
        
        async def generate_text(self, prompt):
            return "Mock response from LLM"
    
    mock_llm = MockLLMService()
    
    # Create agent with mock LLM
    agent = TextToSQLAgent(
        view_service=view_service,
        embedding_service=embedding_service,
        vector_service=vector_service,
        hitl_service=hitl_service,
        llm_service=mock_llm,
        session_service=session_service
    )
    
    # Skip if embedding service is not available
    try:
        test_embedding = await embedding_service.get_embedding("test")
        if not test_embedding:
            pytest.skip("Embedding service not available for full flow test")
    except Exception as e:
        pytest.skip(f"Embedding service not available: {e}")
    
    # Setup test data
    for view in sample_views:
        await view_service.create_view(view)
    
    all_views = await view_service.get_all_views()
    await vector_service.reindex_all_views(all_views, embedding_service)
    
    # Process a query
    result = await agent.process_query("Show me users created in the last 30 days")
    
    # Verify result
    assert result is not None
    assert "session_id" in result
    assert "response" in result
    
    # Verify session was created
    session = await session_service.get_session(result["session_id"])
    assert session is not None


@pytest.mark.asyncio
async def test_error_handling(services):
    """Test error handling in various scenarios."""
    view_service = services["view_service"]
    
    # Test getting non-existent view
    result = await view_service.get_view_by_name("NON_EXISTENT_VIEW")
    assert result is None
    
    # Test deleting non-existent view
    success = await view_service.delete_view("NON_EXISTENT_VIEW")
    assert success is False
    
    # Test HITL with invalid request ID
    hitl_service = services["hitl_service"]
    request = await hitl_service.get_request("invalid-request-id")
    assert request is None
    
    # Test approving non-existent request
    success = await hitl_service.approve_request("invalid-request-id")
    assert success is False


@pytest.mark.asyncio
async def test_cleanup_operations(services):
    """Test cleanup operations."""
    hitl_service = services["hitl_service"]
    session_service = services["session_service"]
    
    # Test HITL cleanup
    await hitl_service.cleanup_expired_requests()
    
    # Test session cleanup
    deleted_count = await session_service.cleanup_old_sessions(days_old=0)
    # Should be >= 0 (may not have any old sessions to delete)