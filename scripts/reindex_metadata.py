"""
Script to reindex all view metadata in OpenSearch using actual application services.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reindex_all():
    """Reindex all view metadata in OpenSearch using actual application services."""
    try:
        # Import and initialize actual application services
        from text_to_sql_rag.services.mongodb_service import MongoDBService
        from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
        from text_to_sql_rag.config.settings import settings
        
        logger.info("🔄 Initializing application services...")
        
        # Initialize MongoDB service
        mongodb_service = MongoDBService()
        if not mongodb_service.is_connected():
            logger.error("❌ MongoDB service not connected")
            return False
        
        logger.info("✅ MongoDB service connected")
        
        # Initialize Vector service  
        vector_service = LlamaIndexVectorService()
        if not vector_service.health_check():
            logger.error("❌ Vector service health check failed")
            return False
            
        logger.info("✅ Vector service initialized")
        
        # Get all documents from MongoDB
        logger.info("🔄 Retrieving documents from MongoDB...")
        documents = list(mongodb_service.get_all_documents())
        
        if not documents:
            logger.warning("No documents found in MongoDB")
            return True
        
        logger.info(f"Found {len(documents)} documents to reindex")
        
        # Get index stats before reindexing
        try:
            before_stats = vector_service.get_index_stats()
            logger.info(f"Index stats before reindexing: {before_stats}")
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
        
        # Reindex all documents
        logger.info("🔄 Starting reindexing process...")
        reindexed_count = 0
        
        for doc in documents:
            try:
                document_id = doc.get("view_name") or doc.get("report_name") or doc.get("lookup_name") or str(doc["_id"])
                content = doc.get("content", "")
                
                if not content:
                    logger.warning(f"No content for document {document_id}")
                    continue
                
                # Use the vector service to add/update the document
                success = vector_service.add_document(
                    document_id=document_id,
                    content=content,
                    metadata=doc.get("metadata", {}),
                    document_type=doc.get("document_type", "unknown")
                )
                
                if success:
                    reindexed_count += 1
                    if reindexed_count % 10 == 0:
                        logger.info(f"Reindexed {reindexed_count}/{len(documents)} documents...")
                else:
                    logger.error(f"Failed to reindex document: {document_id}")
                    
            except Exception as e:
                logger.error(f"Error reindexing document: {e}")
        
        # Get index stats after reindexing
        try:
            after_stats = vector_service.get_index_stats()
            logger.info(f"Index stats after reindexing: {after_stats}")
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
        
        logger.info("=" * 60)
        logger.info("🎉 REINDEXING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 SUMMARY:")
        logger.info(f"   • Total Documents: {len(documents)}")
        logger.info(f"   • Successfully Reindexed: {reindexed_count}")
        logger.info(f"   • Failed: {len(documents) - reindexed_count}")
        logger.info("=" * 60)
        
        # Cleanup
        mongodb_service.close()
        
        return reindexed_count > 0
        
    except Exception as e:
        logger.error(f"❌ Reindexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    success = await reindex_all()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)