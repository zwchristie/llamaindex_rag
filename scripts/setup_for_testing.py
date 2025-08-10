"""
Setup script to prepare the system for automated testing.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.config.new_settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_services():
    """Check if all required services are available."""
    print("🔍 Checking service availability...")
    
    services_ok = True
    
    # Check MongoDB
    try:
        import motor.motor_asyncio
        settings = get_settings()
        
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
        await client.admin.command('ping')
        print("✅ MongoDB: Connected")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB: {e}")
        services_ok = False
    
    # Check OpenSearch
    try:
        from opensearchpy import AsyncOpenSearch
        settings = get_settings()
        
        client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=None,
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
        )
        await client.ping()
        print("✅ OpenSearch: Connected")
        await client.close()
    except Exception as e:
        print(f"❌ OpenSearch: {e}")
        services_ok = False
    
    # Check Bedrock endpoint (optional)
    try:
        import httpx
        settings = get_settings()
        
        # Just verify the endpoint URL is configured
        if settings.bedrock_endpoint_url and "http" in settings.bedrock_endpoint_url:
            print("✅ Bedrock Endpoint: Configured")
        else:
            print("⚠️ Bedrock Endpoint: Not configured (will use mock)")
    except Exception as e:
        print(f"⚠️ Bedrock Endpoint: {e}")
    
    return services_ok


async def verify_database_setup():
    """Verify database collections and indexes."""
    print("\n🔧 Verifying database setup...")
    
    try:
        import motor.motor_asyncio
        from text_to_sql_rag.services.view_service import ViewService
        from text_to_sql_rag.services.session_service import SessionService
        
        settings = get_settings()
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.mongodb_url)
        db = client[settings.mongodb_database]
        
        # Initialize services to create indexes
        view_service = ViewService(db)
        session_service = SessionService(db)
        
        await view_service.ensure_indexes()
        await session_service.ensure_indexes()
        
        print("✅ Database indexes created")
        
        # Check collections
        collections = await db.list_collection_names()
        expected = ["view_metadata", "session_states", "hitl_requests"]
        
        for collection in expected:
            if collection in collections:
                count = await db[collection].count_documents({})
                print(f"✅ Collection '{collection}': {count} documents")
            else:
                print(f"ℹ️ Collection '{collection}': Will be created on first use")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


async def verify_opensearch_setup():
    """Verify OpenSearch setup."""
    print("\n🔍 Verifying OpenSearch setup...")
    
    try:
        from opensearchpy import AsyncOpenSearch
        from text_to_sql_rag.services.embedding_service import VectorService
        
        settings = get_settings()
        client = AsyncOpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_auth=None,
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
        )
        
        # Check cluster health
        health = await client.cluster.health()
        print(f"✅ OpenSearch cluster status: {health.get('status', 'unknown')}")
        
        # Check if our index exists
        index_name = settings.opensearch_index_name
        if await client.indices.exists(index=index_name):
            stats = await client.indices.stats(index=index_name)
            doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
            print(f"✅ Index '{index_name}': {doc_count} documents")
        else:
            print(f"ℹ️ Index '{index_name}': Will be created when needed")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ OpenSearch setup failed: {e}")
        return False


async def test_embedding_service():
    """Test the embedding service."""
    print("\n🧠 Testing embedding service...")
    
    try:
        from text_to_sql_rag.services.embedding_service import EmbeddingService
        
        settings = get_settings()
        embedding_service = EmbeddingService(
            settings.bedrock_endpoint_url,
            settings.bedrock_embedding_model
        )
        
        # Test embedding generation
        test_text = "test embedding generation"
        embedding = await embedding_service.get_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Embedding service: Generated {len(embedding)}-dimensional embedding")
            return True
        else:
            print("⚠️ Embedding service: Using mock embeddings")
            return True  # Mock is okay for testing
            
    except Exception as e:
        print(f"⚠️ Embedding service: {e} (will use mock)")
        return True  # Mock fallback is fine


async def setup_test_data():
    """Setup minimal test data if needed."""
    print("\n📊 Setting up test data...")
    
    try:
        # This will be done by the actual test scripts
        print("ℹ️ Test data will be created by individual tests")
        return True
        
    except Exception as e:
        print(f"❌ Test data setup failed: {e}")
        return False


async def main():
    """Main setup function."""
    print("🚀 SYSTEM TESTING SETUP")
    print("=" * 40)
    print("Preparing system for comprehensive automated testing")
    print()
    
    all_good = True
    
    # Check services
    services_ok = await check_services()
    all_good &= services_ok
    
    if not services_ok:
        print("\n❌ Some services are not available. Please ensure:")
        print("  1. MongoDB is running (docker compose up mongodb)")
        print("  2. OpenSearch is running (docker compose up opensearch)")
        print("  3. Environment variables are set correctly")
        return False
    
    # Verify database setup
    db_ok = await verify_database_setup()
    all_good &= db_ok
    
    # Verify OpenSearch setup
    os_ok = await verify_opensearch_setup()
    all_good &= os_ok
    
    # Test embedding service
    embed_ok = await test_embedding_service()
    all_good &= embed_ok
    
    # Setup test data
    data_ok = await setup_test_data()
    all_good &= data_ok
    
    print("\n" + "=" * 40)
    if all_good:
        print("✅ SETUP COMPLETE - Ready for automated testing!")
        print()
        print("Next steps:")
        print("  1. Run: python tests/run_system_tests.py")
        print("  2. Or: pytest tests/system/ -v -s")
        print()
    else:
        print("❌ SETUP INCOMPLETE - Please fix issues above")
        print()
    
    return all_good


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)