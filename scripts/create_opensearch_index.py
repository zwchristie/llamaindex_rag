#!/usr/bin/env python3
"""
Create OpenSearch index manually for k-NN vector search.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from opensearchpy import AsyncOpenSearch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def create_vector_index():
    """Create OpenSearch index with proper k-NN configuration."""
    try:
        # OpenSearch connection
        opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
        opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        
        opensearch_client = AsyncOpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_auth=None, use_ssl=False, verify_certs=False
        )
        
        # Check connection
        cluster_info = await opensearch_client.info()
        logger.info(f"✅ OpenSearch connected: {cluster_info['version']['number']}")
        
        index_name = "view_metadata"
        
        # Delete existing index if it exists
        if await opensearch_client.indices.exists(index_name):
            logger.info(f"🗑️  Deleting existing index: {index_name}")
            await opensearch_client.indices.delete(index_name)
        
        # Determine vector size
        use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
        vector_size = 1536 if use_mock else 1024
        logger.info(f"📏 Using {vector_size}-dimensional vectors")
        
        # Create index with k-NN settings
        index_config = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "view_name": {
                        "type": "keyword"
                    },
                    "schema": {
                        "type": "keyword"
                    }, 
                    "description": {
                        "type": "text"
                    },
                    "business_context": {
                        "type": "text"
                    },
                    "full_text": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": vector_size,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil"
                        }
                    },
                    "_uploaded_at": {
                        "type": "date"
                    },
                    "_source_file": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        # Create the index
        await opensearch_client.indices.create(index_name, body=index_config)
        logger.info(f"✅ Created OpenSearch index '{index_name}' successfully")
        
        # Verify index creation
        index_info = await opensearch_client.indices.get(index_name)
        settings = index_info[index_name]['settings']
        mappings = index_info[index_name]['mappings']
        
        logger.info(f"📊 Index settings: knn={settings['index'].get('knn', 'false')}")
        logger.info(f"🔍 Vector dimension: {mappings['properties']['embedding']['dimension']}")
        
        await opensearch_client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create index: {e}")
        if 'opensearch_client' in locals():
            await opensearch_client.close()
        return False

async def main():
    """Main function."""
    print("🔧 OPENSEARCH INDEX CREATOR")
    print("="*40)
    
    success = await create_vector_index()
    
    if success:
        print("\n✅ OpenSearch index created successfully!")
        print("You can now run: poetry run python scripts/load_sample_data.py")
    else:
        print("\n❌ Failed to create OpenSearch index")
        print("Check the logs above for details")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)