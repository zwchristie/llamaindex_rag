"""
Script to reindex all view metadata in OpenSearch.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging
from opensearchpy import AsyncOpenSearch
import motor.motor_asyncio
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.services.view_service import ViewService
from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reindex_all():
    """Reindex all view metadata in OpenSearch."""
    try:
        # Get configuration from environment
        mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
        database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
        
        opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
        opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        opensearch_index = os.getenv("OPENSEARCH_INDEX_NAME", "view_metadata")
        vector_field = os.getenv("OPENSEARCH_VECTOR_FIELD", "embedding")
        
        bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT_URL", "https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess")
        embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        use_mock_embeddings = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
        
        logger.info(f"USE_MOCK_EMBEDDINGS environment variable: {os.getenv('USE_MOCK_EMBEDDINGS')}")
        logger.info(f"Using mock embeddings: {use_mock_embeddings}")
        
        # Connect to MongoDB
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
        db = client[database_name]
        view_service = ViewService(db)
        
        logger.info(f"Connected to MongoDB: {database_name}")
        
        # Connect to OpenSearch
        opensearch_client = AsyncOpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_auth=None,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        logger.info(f"Connected to OpenSearch: {opensearch_host}:{opensearch_port}")
        
        # Initialize services
        embedding_service = EmbeddingService(bedrock_endpoint, embedding_model, use_mock=use_mock_embeddings)
        vector_service = VectorService(opensearch_client, opensearch_index, vector_field)
        
        # Get all views from MongoDB
        all_views = await view_service.get_all_views()
        logger.info(f"Found {len(all_views)} views to reindex")
        
        if not all_views:
            logger.warning("No views found in MongoDB. Run seed_mock_data.py first.")
            return
        
        # Reindex all views
        await vector_service.reindex_all_views(all_views, embedding_service)
        
        # Get index stats
        stats = await vector_service.get_index_stats()
        logger.info("Reindexing completed!")
        logger.info(f"Indexed documents: {stats.get('document_count', 0)}")
        logger.info(f"Index size: {stats.get('store_size_mb', 0)} MB")
        
        # Test search functionality
        logger.info("Testing search functionality...")
        test_query = "syndicate participation"
        test_embedding = await embedding_service.get_embedding(test_query)
        
        results = await vector_service.search_similar_views(test_embedding, k=3)
        logger.info(f"Test search for '{test_query}' returned {len(results)} results:")
        for i, (view, score) in enumerate(results[:3], 1):
            logger.info(f"  {i}. {view.view_name} (score: {score:.3f})")
        
        # Close connections
        client.close()
        await opensearch_client.close()
        
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(reindex_all())