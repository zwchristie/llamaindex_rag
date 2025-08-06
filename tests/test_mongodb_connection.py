#!/usr/bin/env python3
"""
MongoDB Connection Test Script

Tests MongoDB connectivity, authentication, and basic CRUD operations.
Can be run independently to validate MongoDB service connectivity.
"""

import os
import sys
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
except ImportError:
    print("❌ ERROR: pymongo not installed. Install with: pip install pymongo")
    sys.exit(1)

# Import our MongoDB service
try:
    from text_to_sql_rag.services.mongodb_service import MongoDBService
    from text_to_sql_rag.config.settings import settings
except ImportError as e:
    print(f"❌ ERROR: Cannot import application modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class MongoDBConnectionTest:
    """Test MongoDB connectivity and operations."""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.test_collection: Optional[Collection] = None
        self.results = []
    
    def log_result(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results.append(result)
        print(f"{status}: {test_name} - {message}")
        if details and not success:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def test_connection_parameters(self):
        """Test connection parameters from configuration."""
        try:
            mongo_url = settings.mongodb.url
            database_name = settings.mongodb.database
            
            if not mongo_url:
                self.log_result(
                    "Connection Parameters", 
                    False, 
                    "MongoDB URL not configured",
                    {"url": mongo_url}
                )
                return False
            
            if not database_name:
                self.log_result(
                    "Connection Parameters", 
                    False, 
                    "Database name not configured",
                    {"database": database_name}
                )
                return False
            
            self.log_result(
                "Connection Parameters", 
                True, 
                f"Configuration loaded successfully",
                {
                    "url": mongo_url,
                    "database": database_name
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Connection Parameters", 
                False, 
                f"Failed to load configuration: {str(e)}"
            )
            return False
    
    def test_basic_connection(self):
        """Test basic MongoDB connection."""
        try:
            mongo_url = settings.mongodb.url
            
            # Create client with short timeout for testing
            self.client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection with ping
            start_time = time.time()
            self.client.admin.command('ping')
            connection_time = time.time() - start_time
            
            self.log_result(
                "Basic Connection", 
                True, 
                f"Connected successfully",
                {
                    "connection_time_ms": round(connection_time * 1000, 2),
                    "server_info": str(self.client.server_info().get('version', 'unknown'))
                }
            )
            return True
            
        except ConnectionFailure as e:
            self.log_result(
                "Basic Connection", 
                False, 
                f"Connection failed: {str(e)}",
                {"error_type": "ConnectionFailure"}
            )
            return False
        except ServerSelectionTimeoutError as e:
            self.log_result(
                "Basic Connection", 
                False, 
                f"Server selection timeout: {str(e)}",
                {"error_type": "ServerSelectionTimeoutError"}
            )
            return False
        except Exception as e:
            self.log_result(
                "Basic Connection", 
                False, 
                f"Unexpected error: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_database_access(self):
        """Test database access and permissions."""
        if not self.client:
            self.log_result("Database Access", False, "No client connection available")
            return False
        
        try:
            database_name = settings.mongodb.database
            self.db = self.client[database_name]
            
            # Test database access by listing collections
            collections = self.db.list_collection_names()
            
            self.log_result(
                "Database Access", 
                True, 
                f"Database accessible",
                {
                    "database": database_name,
                    "collection_count": len(collections),
                    "collections": collections[:5]  # Show first 5
                }
            )
            return True
            
        except OperationFailure as e:
            self.log_result(
                "Database Access", 
                False, 
                f"Database operation failed: {str(e)}",
                {"error_type": "OperationFailure"}
            )
            return False
        except Exception as e:
            self.log_result(
                "Database Access", 
                False, 
                f"Unexpected error: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_collection_operations(self):
        """Test collection creation and basic operations."""
        if not self.db:
            self.log_result("Collection Operations", False, "No database connection available")
            return False
        
        try:
            # Use a test collection
            test_collection_name = "connection_test_collection"
            self.test_collection = self.db[test_collection_name]
            
            # Clean up any existing test documents
            self.test_collection.delete_many({"test_document": True})
            
            # Test document insertion
            test_doc = {
                "test_document": True,
                "timestamp": datetime.utcnow(),
                "test_data": "MongoDB connection test",
                "test_id": "mongodb_test_001"
            }
            
            result = self.test_collection.insert_one(test_doc)
            
            if not result.inserted_id:
                self.log_result("Collection Operations", False, "Document insertion failed")
                return False
            
            # Test document retrieval
            retrieved_doc = self.test_collection.find_one({"test_id": "mongodb_test_001"})
            
            if not retrieved_doc:
                self.log_result("Collection Operations", False, "Document retrieval failed")
                return False
            
            # Test document update
            update_result = self.test_collection.update_one(
                {"test_id": "mongodb_test_001"},
                {"$set": {"updated": True, "update_timestamp": datetime.utcnow()}}
            )
            
            if update_result.modified_count != 1:
                self.log_result("Collection Operations", False, "Document update failed")
                return False
            
            # Test document counting
            count = self.test_collection.count_documents({"test_document": True})
            
            self.log_result(
                "Collection Operations", 
                True, 
                f"CRUD operations successful",
                {
                    "inserted_id": str(result.inserted_id),
                    "documents_found": count,
                    "update_count": update_result.modified_count
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Collection Operations", 
                False, 
                f"Collection operations failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_indexes(self):
        """Test index creation and management."""
        if not self.test_collection:
            self.log_result("Index Operations", False, "No test collection available")
            return False
        
        try:
            # Create a test index
            index_name = "test_timestamp_index"
            self.test_collection.create_index("timestamp", name=index_name)
            
            # List indexes
            indexes = list(self.test_collection.list_indexes())
            index_names = [idx.get("name", "") for idx in indexes]
            
            if index_name not in index_names:
                self.log_result("Index Operations", False, "Index creation failed")
                return False
            
            self.log_result(
                "Index Operations", 
                True, 
                f"Index operations successful",
                {
                    "indexes_created": 1,
                    "total_indexes": len(indexes),
                    "index_names": index_names
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Index Operations", 
                False, 
                f"Index operations failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_mongodb_service(self):
        """Test our custom MongoDB service."""
        try:
            service = MongoDBService()
            
            # Test connection status
            is_connected = service.is_connected()
            
            if not is_connected:
                self.log_result(
                    "MongoDB Service", 
                    False, 
                    "MongoDB service reports not connected"
                )
                return False
            
            # Test health check
            health = service.health_check()
            
            if health.get("status") != "healthy":
                self.log_result(
                    "MongoDB Service", 
                    False, 
                    f"MongoDB service health check failed: {health.get('status')}",
                    {"health_details": health}
                )
                return False
            
            # Test collection stats
            stats = service.get_collection_stats()
            
            self.log_result(
                "MongoDB Service", 
                True, 
                f"MongoDB service working correctly",
                {
                    "connection_status": is_connected,
                    "health_status": health.get("status"),
                    "has_stats": "error" not in stats
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "MongoDB Service", 
                False, 
                f"MongoDB service test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def cleanup(self):
        """Clean up test data and connections."""
        try:
            if self.test_collection:
                # Remove test documents
                self.test_collection.delete_many({"test_document": True})
                
                # Drop test indexes
                try:
                    self.test_collection.drop_index("test_timestamp_index")
                except:
                    pass  # Index might not exist
            
            if self.client:
                self.client.close()
                
            self.log_result("Cleanup", True, "Test cleanup completed successfully")
            
        except Exception as e:
            self.log_result(
                "Cleanup", 
                False, 
                f"Cleanup failed: {str(e)}"
            )
    
    def run_all_tests(self):
        """Run all MongoDB tests."""
        print("🔍 Starting MongoDB Connection Tests")
        print("=" * 50)
        
        # Test sequence
        tests = [
            self.test_connection_parameters,
            self.test_basic_connection,
            self.test_database_access,
            self.test_collection_operations,
            self.test_indexes,
            self.test_mongodb_service
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ FAIL: {test.__name__} - Unexpected error: {str(e)}")
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All MongoDB tests passed!")
        else:
            print(f"⚠️  {total - passed} test(s) failed")
        
        # Always run cleanup
        self.cleanup()
        
        return passed == total


def main():
    """Main function to run MongoDB connection tests."""
    
    print("🧪 MongoDB Connection Test Suite")
    print("This script tests MongoDB connectivity and basic operations")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/text_to_sql_rag"):
        print("❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run tests
    tester = MongoDBConnectionTest()
    success = tester.run_all_tests()
    
    # Print configuration help
    print("\n" + "=" * 50)
    print("📝 Configuration Notes:")
    print(f"   MongoDB URL: {settings.mongodb.url}")
    print(f"   Database: {settings.mongodb.database}")
    print()
    print("   To configure MongoDB connection:")
    print("   - Set MONGODB_URL environment variable")
    print("   - Set MONGODB_DATABASE environment variable")
    print("   - Or update src/text_to_sql_rag/config/settings.py")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())