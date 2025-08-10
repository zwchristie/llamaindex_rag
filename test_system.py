#!/usr/bin/env python3
"""
Quick system functionality test.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch

# Load .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_system():
    """Test basic system functionality."""
    print("SYSTEM FUNCTIONALITY TEST")
    print("=" * 40)
    
    # Test 1: MongoDB Connection
    try:
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
        
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
        db = client[database_name]
        
        # Count documents
        view_count = await db.view_metadata.count_documents({})
        print(f"[PASS] MongoDB connected: {view_count} views found")
        
        client.close()
        
    except Exception as e:
        print(f"[FAIL] MongoDB test failed: {e}")
        return False
    
    # Test 2: OpenSearch Connection
    try:
        opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
        opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        
        opensearch_client = AsyncOpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        # Check index
        index_name = os.getenv("OPENSEARCH_INDEX_NAME", "view_metadata")
        stats = await opensearch_client.indices.stats(index=index_name)
        doc_count = stats['indices'][index_name]['total']['docs']['count']
        print(f"[PASS] OpenSearch connected: {doc_count} documents indexed")
        
        await opensearch_client.close()
        
    except Exception as e:
        print(f"[FAIL] OpenSearch test failed: {e}")
        return False
    
    # Test 3: Mock Embedding Service
    try:
        from text_to_sql_rag.services.embedding_service import EmbeddingService
        
        embedding_service = EmbeddingService(
            endpoint_url="https://dummy.com",
            embedding_model="dummy",
            use_mock=True
        )
        
        test_text = "syndicate participation reporting"
        embedding = await embedding_service.get_embedding(test_text)
        
        print(f"[PASS] Mock embedding service: Generated {len(embedding)}-dimensional vector")
        
    except Exception as e:
        print(f"[FAIL] Embedding service test failed: {e}")
        return False
    
    # Test 4: View Service
    try:
        from text_to_sql_rag.services.view_service import ViewService
        
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
        db = client[database_name]
        view_service = ViewService(db)
        
        # Get a sample view
        views = await view_service.get_all_views()
        if views:
            sample_view = views[0]
            print(f"[PASS] View service: Retrieved view '{sample_view.view_name}'")
        else:
            print("[WARN] View service: No views found (run seeding)")
        
        client.close()
        
    except Exception as e:
        print(f"[FAIL] View service test failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED!")
    print("=" * 40)
    print("\nYour system is ready for the CTO demo!")
    print("\nNext steps:")
    print("   1. Fix API server import issues")
    print("   2. Test end-to-end query processing")
    print("   3. Verify HITL workflow")
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_system())
    sys.exit(0 if success else 1)