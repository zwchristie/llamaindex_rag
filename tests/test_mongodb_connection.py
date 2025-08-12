#!/usr/bin/env python3
"""
Test MongoDB connectivity using actual application services.
Tests the real connection and functionality, not separate test connections.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MongoDBConnectionTest:
    """Test MongoDB using actual application services."""
    
    def __init__(self):
        self.mongodb_service = None
        self.results = []
    
    def log_result(self, test_name: str, success: bool, message: str, details: dict = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {}
        }
        self.results.append(result)
        print(f"{status}: {test_name} - {message}")
        if details and not success:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def test_import_services(self):
        """Test importing actual application services."""
        try:
            from text_to_sql_rag.config.settings import settings
            from text_to_sql_rag.services.mongodb_service import MongoDBService
            
            self.settings = settings
            
            self.log_result(
                "Import Services",
                True,
                "Successfully imported application services"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Import Services",
                False,
                f"Failed to import services: {e}"
            )
            return False
    
    def test_configuration(self):
        """Test MongoDB configuration."""
        try:
            # Check configuration
            mongodb_url = self.settings.mongodb.url
            database_name = self.settings.mongodb.database
            
            if not mongodb_url:
                self.log_result(
                    "Configuration",
                    False,
                    "MONGODB_URL not configured",
                    {"mongodb_url": mongodb_url}
                )
                return False
            
            self.log_result(
                "Configuration",
                True,
                f"Configuration loaded successfully",
                {
                    "mongodb_url": mongodb_url.split('@')[1] if '@' in mongodb_url else mongodb_url,  # Hide credentials
                    "database": database_name
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Configuration",
                False,
                f"Configuration error: {e}"
            )
            return False
    
    def test_service_initialization(self):
        """Test MongoDB service initialization."""
        try:
            from text_to_sql_rag.services.mongodb_service import MongoDBService
            
            self.mongodb_service = MongoDBService()
            
            self.log_result(
                "Service Initialization",
                True,
                f"MongoDB service initialized successfully"
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Service Initialization",
                False,
                f"Service initialization failed: {e}"
            )
            return False
    
    def test_connection_check(self):
        """Test MongoDB connection status."""
        try:
            is_connected = self.mongodb_service.is_connected()
            
            if is_connected:
                self.log_result(
                    "Connection Check",
                    True,
                    "MongoDB connection is active"
                )
            else:
                self.log_result(
                    "Connection Check",
                    False,
                    "MongoDB connection is not active"
                )
            
            return is_connected
            
        except Exception as e:
            self.log_result(
                "Connection Check",
                False,
                f"Connection check failed: {e}"
            )
            return False
    
    def test_health_check(self):
        """Test MongoDB health check."""
        try:
            health = self.mongodb_service.health_check()
            
            is_healthy = health.get("status") == "healthy"
            
            self.log_result(
                "Health Check",
                is_healthy,
                f"Health check {'passed' if is_healthy else 'failed'}",
                health
            )
            
            return is_healthy
            
        except Exception as e:
            self.log_result(
                "Health Check",
                False,
                f"Health check error: {e}"
            )
            return False
    
    def test_document_operations(self):
        """Test basic document operations (CRUD)."""
        try:
            # Test document creation
            test_doc = {
                "document_type": "test",
                "test_id": "mongodb_connection_test",
                "name": "Connection Test Document",
                "description": "Test document for MongoDB connection validation",
                "created_at": datetime.utcnow(),
                "test_data": {"key": "value", "number": 42}
            }
            
            # Insert document
            doc_id = self.mongodb_service.store_document(test_doc)
            
            if not doc_id:
                self.log_result(
                    "Document Operations",
                    False,
                    "Failed to insert test document"
                )
                return False
            
            # Retrieve document
            retrieved_doc = self.mongodb_service.get_document_by_id(doc_id)
            
            if not retrieved_doc:
                self.log_result(
                    "Document Operations",
                    False,
                    "Failed to retrieve test document"
                )
                return False
            
            # Update document
            update_data = {"updated_at": datetime.utcnow(), "test_updated": True}
            update_success = self.mongodb_service.update_document(doc_id, update_data)
            
            if not update_success:
                self.log_result(
                    "Document Operations",
                    False,
                    "Failed to update test document"
                )
                return False
            
            # Verify update
            updated_doc = self.mongodb_service.get_document_by_id(doc_id)
            if not updated_doc or not updated_doc.get("test_updated"):
                self.log_result(
                    "Document Operations",
                    False,
                    "Document update verification failed"
                )
                return False
            
            # Delete document
            delete_success = self.mongodb_service.delete_document(doc_id)
            
            if not delete_success:
                self.log_result(
                    "Document Operations",
                    False,
                    "Failed to delete test document"
                )
                return False
            
            # Verify deletion
            deleted_doc = self.mongodb_service.get_document_by_id(doc_id)
            if deleted_doc:
                self.log_result(
                    "Document Operations",
                    False,
                    "Document deletion verification failed"
                )
                return False
            
            self.log_result(
                "Document Operations",
                True,
                "All CRUD operations completed successfully",
                {
                    "operations": "CREATE, READ, UPDATE, DELETE",
                    "test_doc_id": str(doc_id)
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Document Operations",
                False,
                f"Document operations failed: {e}"
            )
            return False
    
    def test_collection_stats(self):
        """Test collection statistics retrieval."""
        try:
            stats = self.mongodb_service.get_collection_stats()
            
            if "error" in stats:
                self.log_result(
                    "Collection Stats",
                    False,
                    f"Failed to get collection stats: {stats['error']}"
                )
                return False
            
            self.log_result(
                "Collection Stats",
                True,
                "Successfully retrieved collection statistics",
                {
                    "document_count": stats.get("count", "N/A"),
                    "avg_obj_size": stats.get("avgObjSize", "N/A"),
                    "storage_size": stats.get("storageSize", "N/A")
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Collection Stats",
                False,
                f"Collection stats failed: {e}"
            )
            return False
    
    def test_query_operations(self):
        """Test query operations."""
        try:
            # Insert test documents
            test_docs = [
                {
                    "document_type": "test_query",
                    "name": f"Test Document {i}",
                    "category": "query_test",
                    "value": i * 10,
                    "created_at": datetime.utcnow()
                }
                for i in range(3)
            ]
            
            inserted_ids = []
            for doc in test_docs:
                doc_id = self.mongodb_service.store_document(doc)
                if doc_id:
                    inserted_ids.append(doc_id)
            
            if len(inserted_ids) != 3:
                self.log_result(
                    "Query Operations",
                    False,
                    f"Failed to insert all test documents: {len(inserted_ids)}/3"
                )
                return False
            
            # Test query by document type
            query_results = list(self.mongodb_service.get_documents_by_type("test_query"))
            
            if len(query_results) < 3:
                self.log_result(
                    "Query Operations",
                    False,
                    f"Query returned insufficient results: {len(query_results)}"
                )
                return False
            
            # Test complex query
            complex_query = {"category": "query_test", "value": {"$gte": 10}}
            complex_results = list(self.mongodb_service.documents_collection.find(complex_query))
            
            # Cleanup test documents
            for doc_id in inserted_ids:
                self.mongodb_service.delete_document(doc_id)
            
            self.log_result(
                "Query Operations",
                True,
                "Query operations completed successfully",
                {
                    "simple_query_results": len(query_results),
                    "complex_query_results": len(complex_results),
                    "cleanup_completed": True
                }
            )
            return True
            
        except Exception as e:
            # Cleanup on error
            if 'inserted_ids' in locals():
                for doc_id in inserted_ids:
                    try:
                        self.mongodb_service.delete_document(doc_id)
                    except:
                        pass
            
            self.log_result(
                "Query Operations",
                False,
                f"Query operations failed: {e}"
            )
            return False
    
    def run_all_tests(self):
        """Run all MongoDB tests."""
        print("🧪 Starting MongoDB Connection Tests")
        print("=" * 50)
        
        tests = [
            self.test_import_services,
            self.test_configuration,
            self.test_service_initialization,
            self.test_connection_check,
            self.test_health_check,
            self.test_document_operations,
            self.test_collection_stats,
            self.test_query_operations
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ FAIL: {test.__name__} - Unexpected error: {e}")
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All MongoDB tests passed!")
        else:
            print(f"⚠️  {total - passed} test(s) failed")
        
        print("\n" + "=" * 50)
        print("📝 Configuration Notes:")
        print(f"   MongoDB URL: {self.settings.mongodb.url.split('@')[1] if '@' in self.settings.mongodb.url else self.settings.mongodb.url}")
        print(f"   Database: {self.settings.mongodb.database}")
        print()
        print("   To configure MongoDB connection:")
        print("   - Set MONGODB_URL environment variable")
        print("   - Set MONGODB_DATABASE environment variable")
        
        return passed == total


def main():
    """Main function to run MongoDB connection tests."""
    print("🧪 MongoDB Connection Test Suite")
    print("This tests the actual MongoDB functionality using application services")
    print()
    
    tester = MongoDBConnectionTest()
    success = tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)