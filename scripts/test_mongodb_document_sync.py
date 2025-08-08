#!/usr/bin/env python3
"""
Test script to validate MongoDB document retrieval and Vector Store synchronization.

This script tests the new MongoDB-to-Vector-Store synchronization process
that replaces the old file-based sync approach.

Usage:
    python scripts/test_mongodb_document_sync.py
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, List
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.services.mongodb_service import MongoDBService
from text_to_sql_rag.services.vector_service import LlamaIndexVectorService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_mongodb_connection():
    """Test MongoDB connection and document retrieval."""
    logger.info("🔧 Testing MongoDB connection...")
    
    try:
        mongodb_service = MongoDBService()
        
        if not mongodb_service.is_connected():
            logger.error("❌ MongoDB connection failed")
            return False
        
        logger.info("✅ MongoDB connected successfully")
        
        # Test health check
        health = mongodb_service.health_check()
        logger.info(f"MongoDB Health: {health}")
        
        return mongodb_service
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection test failed: {e}")
        return False


def test_document_retrieval(mongodb_service: MongoDBService):
    """Test document retrieval from MongoDB."""
    logger.info("📄 Testing document retrieval...")
    
    try:
        # Get all documents
        all_docs = mongodb_service.get_all_documents()
        logger.info(f"Retrieved {len(all_docs)} documents from MongoDB")
        
        if len(all_docs) == 0:
            logger.warning("⚠️  No documents found in MongoDB")
            logger.info("Run the discover_and_migrate_metadata.py script first to populate MongoDB")
            return []
        
        # Show document types
        doc_types = {}
        for doc in all_docs:
            doc_type = doc.get("document_type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        logger.info("Document types found:")
        for doc_type, count in doc_types.items():
            logger.info(f"  {doc_type}: {count} documents")
        
        # Show sample document structure
        if all_docs:
            sample_doc = all_docs[0]
            logger.info("Sample document structure:")
            for key in sample_doc.keys():
                if key == "content":
                    logger.info(f"  {key}: <{len(sample_doc[key])} characters>")
                else:
                    logger.info(f"  {key}: {sample_doc[key]}")
        
        return all_docs
        
    except Exception as e:
        logger.error(f"❌ Document retrieval test failed: {e}")
        return []


def test_vector_service_connection():
    """Test Vector Service connection."""
    logger.info("🔍 Testing Vector Service connection...")
    
    try:
        vector_service = LlamaIndexVectorService()
        
        if not vector_service.health_check():
            logger.error("❌ Vector Service health check failed")
            return False
        
        logger.info("✅ Vector Service connected successfully")
        
        # Get index stats
        stats = vector_service.get_index_stats()
        logger.info(f"Vector Store Stats: {stats}")
        
        return vector_service
        
    except Exception as e:
        logger.error(f"❌ Vector Service connection test failed: {e}")
        return False


def test_document_embedding(mongodb_service: MongoDBService, vector_service: LlamaIndexVectorService, max_test_docs: int = 3):
    """Test embedding documents from MongoDB to Vector Store."""
    logger.info("🚀 Testing document embedding process...")
    
    try:
        # Get a few documents to test
        mongo_docs = mongodb_service.get_all_documents()[:max_test_docs]
        
        if not mongo_docs:
            logger.warning("⚠️  No documents available for embedding test")
            return False
        
        logger.info(f"Testing embedding for {len(mongo_docs)} documents")
        
        success_count = 0
        for mongo_doc in mongo_docs:
            try:
                document_id = mongo_doc.get("schema_name", mongo_doc.get("file_path", "test_doc"))
                
                logger.info(f"Embedding document: {document_id}")
                
                # Prepare metadata for vector store
                vector_metadata = {
                    **mongo_doc.get("metadata", {}),
                    "content_hash": mongo_doc.get("content_hash"),
                    "updated_at": mongo_doc.get("updated_at"),
                    "file_path": mongo_doc.get("file_path"),
                    "document_type": mongo_doc.get("document_type"),
                    "catalog": mongo_doc.get("catalog"),
                    "schema_name": mongo_doc.get("schema_name")
                }
                
                # Add document to vector store
                success = vector_service.add_document(
                    document_id=document_id,
                    content=mongo_doc["content"],
                    metadata=vector_metadata,
                    document_type=mongo_doc["document_type"]
                )
                
                if success:
                    success_count += 1
                    logger.info(f"✅ Successfully embedded: {document_id}")
                    
                    # Test retrieval
                    doc_info = vector_service.get_document_info(document_id)
                    logger.info(f"Document info: {doc_info}")
                else:
                    logger.error(f"❌ Failed to embed: {document_id}")
                
            except Exception as e:
                logger.error(f"❌ Error embedding document: {e}")
        
        logger.info(f"Embedding test completed: {success_count}/{len(mongo_docs)} successful")
        return success_count == len(mongo_docs)
        
    except Exception as e:
        logger.error(f"❌ Document embedding test failed: {e}")
        return False


def test_vector_search(vector_service: LlamaIndexVectorService):
    """Test vector search functionality."""
    logger.info("🔍 Testing vector search...")
    
    try:
        test_queries = [
            "deal information",
            "tranche details", 
            "business domain data",
            "view metadata"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            
            results = vector_service.search_similar(
                query=query,
                retriever_type="hybrid",
                similarity_top_k=3
            )
            
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                logger.info(f"  Result {i+1}: {result.get('metadata', {}).get('document_type', 'unknown')} - {result.get('score', 0):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector search test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("🧪 Starting comprehensive MongoDB-to-Vector synchronization test")
    logger.info("=" * 70)
    
    # Test 1: MongoDB Connection
    mongodb_service = test_mongodb_connection()
    if not mongodb_service:
        logger.error("❌ Stopping tests - MongoDB connection failed")
        return False
    
    # Test 2: Document Retrieval
    mongo_docs = test_document_retrieval(mongodb_service)
    if not mongo_docs:
        logger.warning("⚠️  No documents to test with")
    
    # Test 3: Vector Service Connection
    vector_service = test_vector_service_connection()
    if not vector_service:
        logger.error("❌ Stopping tests - Vector Service connection failed")
        return False
    
    # Test 4: Document Embedding
    if mongo_docs:
        embedding_success = test_document_embedding(mongodb_service, vector_service)
        if not embedding_success:
            logger.warning("⚠️  Document embedding had issues")
    
    # Test 5: Vector Search
    search_success = test_vector_search(vector_service)
    if not search_success:
        logger.warning("⚠️  Vector search had issues")
    
    # Test 6: Full Startup Sync Simulation
    logger.info("🔄 Testing full startup sync simulation...")
    start_time = time.time()
    
    embedded_count = 0
    error_count = 0
    
    for mongo_doc in mongo_docs:
        try:
            document_id = mongo_doc.get("schema_name", mongo_doc.get("file_path", "unknown"))
            
            # Check if document needs embedding (simulate startup logic)
            doc_info = vector_service.get_document_info(document_id)
            
            mongo_content_hash = mongo_doc.get("content_hash")
            vector_content_hash = doc_info.get("metadata", {}).get("content_hash")
            
            if (doc_info.get("status") == "not_found" or 
                mongo_content_hash != vector_content_hash):
                
                # Would embed here in real startup
                logger.info(f"Would embed: {document_id}")
                embedded_count += 1
            else:
                logger.info(f"Up to date: {document_id}")
                
        except Exception as e:
            error_count += 1
            logger.error(f"Error in startup sync simulation: {e}")
    
    duration = time.time() - start_time
    
    logger.info("📊 Test Results Summary")
    logger.info("=" * 70)
    logger.info(f"✅ MongoDB Connection: {'PASS' if mongodb_service else 'FAIL'}")
    logger.info(f"✅ Document Retrieval: {'PASS' if mongo_docs else 'FAIL'}")
    logger.info(f"✅ Vector Service: {'PASS' if vector_service else 'FAIL'}")
    logger.info(f"✅ Documents in MongoDB: {len(mongo_docs) if mongo_docs else 0}")
    logger.info(f"✅ Startup Sync Simulation: {embedded_count} would be embedded, {error_count} errors")
    logger.info(f"✅ Test Duration: {duration:.2f} seconds")
    
    if mongodb_service:
        mongodb_service.close_connection()
    
    return True


if __name__ == "__main__":
    print("🧪 MongoDB Document Synchronization Test")
    print("=" * 50)
    print("This script tests the new MongoDB-to-Vector-Store sync process.")
    print()
    
    try:
        success = run_comprehensive_test()
        
        if success:
            print("\n🎉 All tests completed!")
            print("\nNext steps:")
            print("  1. If documents were found, the sync process is working")
            print("  2. If no documents found, run: python scripts/discover_and_migrate_metadata.py")
            print("  3. Start your application to test the full startup process")
        else:
            print("\n❌ Some tests failed!")
            print("Check the error messages above for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)