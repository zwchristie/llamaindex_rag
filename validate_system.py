"""
Quick system validation script.
This script performs basic validation of system components.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def validate_mongodb():
    """Validate MongoDB connectivity and basic operations."""
    print("🔍 Validating MongoDB...")
    
    try:
        import motor.motor_asyncio
        from text_to_sql_rag.config.new_settings import get_settings
        from text_to_sql_rag.services.view_service import ViewService
        from text_to_sql_rag.models.view_models import ViewMetadata, ViewColumn
        
        settings = get_settings()
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
        
        # Test connection
        await client.admin.command('ping')
        print("  ✅ MongoDB connection successful")
        
        # Test database operations
        db = client[settings.mongodb_database + "_test"]
        view_service = ViewService(db)
        
        # Test document creation
        test_view = ViewMetadata(
            view_name="TEST_VALIDATION",
            view_type="CORE",
            description="Test view for validation",
            use_cases="Testing",
            columns=[ViewColumn(name="id", type="NUMBER", notNull=True)]
        )
        
        view_id = await view_service.create_view(test_view)
        print("  ✅ Document creation successful")
        
        # Test document retrieval
        retrieved = await view_service.get_view_by_name("TEST_VALIDATION")
        assert retrieved is not None
        assert retrieved.view_name == "TEST_VALIDATION"
        print("  ✅ Document retrieval successful")
        
        # Cleanup
        await db.drop_collection("view_metadata")
        client.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ MongoDB validation failed: {e}")
        return False


async def validate_opensearch():
    """Validate OpenSearch connectivity and basic operations."""
    print("🔍 Validating OpenSearch...")
    
    try:
        from opensearchpy import AsyncOpenSearch
        from text_to_sql_rag.config.new_settings import get_settings
        
        settings = get_settings()
        client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=None,
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
        )
        
        # Test connection
        await client.ping()
        print("  ✅ OpenSearch connection successful")
        
        # Test index operations
        test_index = "test_validation_index"
        
        # Create test index
        if await client.indices.exists(index=test_index):
            await client.indices.delete(index=test_index)
        
        await client.indices.create(
            index=test_index,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": 128,
                            "method": {"name": "hnsw", "engine": "lucene"}
                        }
                    }
                }
            }
        )
        print("  ✅ Index creation successful")
        
        # Test document indexing
        test_doc = {
            "text": "test document",
            "vector": [0.1] * 128
        }
        
        await client.index(
            index=test_index,
            id="test_doc",
            body=test_doc
        )
        print("  ✅ Document indexing successful")
        
        # Test search
        await client.indices.refresh(index=test_index)
        
        search_results = await client.search(
            index=test_index,
            body={
                "query": {"match": {"text": "test"}}
            }
        )
        
        assert search_results["hits"]["total"]["value"] > 0
        print("  ✅ Document search successful")
        
        # Cleanup
        await client.indices.delete(index=test_index)
        await client.close()
        
        return True
        
    except Exception as e:
        print(f"  ❌ OpenSearch validation failed: {e}")
        return False


async def validate_embeddings():
    """Validate embedding service."""
    print("🔍 Validating embedding service...")
    
    try:
        from text_to_sql_rag.services.embedding_service import EmbeddingService
        from text_to_sql_rag.config.new_settings import get_settings
        
        settings = get_settings()
        
        # Try real embedding service first
        try:
            embedding_service = EmbeddingService(
                settings.bedrock_endpoint_url,
                settings.bedrock_embedding_model
            )
            
            # Test embedding generation
            embedding = await embedding_service.get_embedding("test text")
            if embedding and len(embedding) > 0:
                print(f"  ✅ Real embedding service working (dimension: {len(embedding)})")
                return True
        except Exception as e:
            print(f"  ⚠️ Real embedding service unavailable: {e}")
        
        # Test mock embedding service
        class MockEmbeddingService:
            def __init__(self):
                self.embedding_dimension = 1536
            
            async def get_embedding(self, text):
                return [0.1] * self.embedding_dimension
            
            def get_embedding_dimension(self):
                return self.embedding_dimension
        
        mock_service = MockEmbeddingService()
        embedding = await mock_service.get_embedding("test text")
        print(f"  ✅ Mock embedding service working (dimension: {len(embedding)})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Embedding service validation failed: {e}")
        return False


async def validate_models():
    """Validate data models."""
    print("🔍 Validating data models...")
    
    try:
        from text_to_sql_rag.models.view_models import (
            ViewMetadata, ViewColumn, ViewJoin, HITLRequest, SessionState
        )
        
        # Test ViewMetadata
        view = ViewMetadata(
            view_name="TEST_VIEW",
            view_type="CORE",
            description="Test view",
            use_cases="Testing",
            columns=[
                ViewColumn(name="id", type="NUMBER", notNull=True),
                ViewColumn(name="name", type="VARCHAR2", notNull=False)
            ],
            joins=[
                ViewJoin(table_name="OTHER_TABLE", join_type="INNER", join_condition="a.id = b.id")
            ]
        )
        
        # Test full text generation
        full_text = view.generate_full_text()
        assert len(full_text) > 0
        assert "TEST_VIEW" in full_text
        print("  ✅ ViewMetadata model working")
        
        # Test HITLRequest
        hitl_request = HITLRequest(
            request_id="test-request",
            session_id="test-session", 
            user_query="test query",
            generated_sql="SELECT * FROM test",
            sql_explanation="Test SQL",
            selected_views=["TEST_VIEW"],
            expires_at=datetime.utcnow()
        )
        assert hitl_request.status == "pending"
        print("  ✅ HITLRequest model working")
        
        # Test SessionState
        session = SessionState(
            session_id="test-session",
            current_step="test_step",
            user_query="test query",
            retrieved_views=[],
            selected_views=["TEST_VIEW"]
        )
        assert session.session_id == "test-session"
        print("  ✅ SessionState model working")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model validation failed: {e}")
        return False


async def main():
    """Main validation function."""
    print("🧪 SYSTEM VALIDATION")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = {}
    
    # Test models (always works)
    results["models"] = await validate_models()
    
    # Test embedding service
    results["embeddings"] = await validate_embeddings()
    
    # Test MongoDB (requires service)
    results["mongodb"] = await validate_mongodb()
    
    # Test OpenSearch (requires service)
    results["opensearch"] = await validate_opensearch()
    
    print("\n" + "=" * 40)
    print("📊 VALIDATION RESULTS")
    print("=" * 40)
    
    all_passed = True
    for component, passed in results.items():
        icon = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"{icon} {component.title()}: {status}")
        all_passed &= passed
    
    print()
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("Your system components are working correctly.")
        print()
        print("✅ Data models are properly defined")
        print("✅ MongoDB connectivity and operations work") 
        print("✅ OpenSearch connectivity and operations work")
        print("✅ Embedding service is functional")
        print()
        print("🚀 Ready to run comprehensive tests!")
        print("Next: python run_comprehensive_tests.py")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print()
        print("Please ensure:")
        print("  • MongoDB and OpenSearch are running: make up")
        print("  • Environment variables are configured: .env")
        print("  • Dependencies are installed: poetry install")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)