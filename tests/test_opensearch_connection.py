#!/usr/bin/env python3
"""
OpenSearch Connection Test Script

Tests OpenSearch connectivity, authentication, and basic operations.
Can be run independently to validate OpenSearch service connectivity.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import requests
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from opensearchpy.exceptions import ConnectionError, RequestError, AuthenticationException
except ImportError as e:
    print(f"❌ ERROR: Required packages not installed: {e}")
    print("Install with: pip install opensearch-py requests")
    sys.exit(1)

# Import our configuration
try:
    from text_to_sql_rag.config.settings import settings
except ImportError as e:
    print(f"❌ ERROR: Cannot import application modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class OpenSearchConnectionTest:
    """Test OpenSearch connectivity and operations."""
    
    def __init__(self):
        self.client: Optional[OpenSearch] = None
        self.test_index = "connection_test_index"
        self.results = []
    
    def log_result(self, test_name: str, success: bool, message: str, details: Optional[Dict] = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.results.append(result)
        print(f"{status}: {test_name} - {message}")
        if details and not success:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def test_configuration(self):
        """Test OpenSearch configuration parameters."""
        try:
            host = settings.opensearch.host
            port = settings.opensearch.port
            use_ssl = settings.opensearch.use_ssl
            username = settings.opensearch.username
            password = settings.opensearch.password
            
            if not host:
                self.log_result(
                    "Configuration", 
                    False, 
                    "OpenSearch host not configured"
                )
                return False
            
            if not port:
                self.log_result(
                    "Configuration", 
                    False, 
                    "OpenSearch port not configured"
                )
                return False
            
            protocol = "https" if use_ssl else "http"
            endpoint = f"{protocol}://{host}:{port}"
            
            self.log_result(
                "Configuration", 
                True, 
                f"Configuration loaded successfully",
                {
                    "endpoint": endpoint,
                    "use_ssl": use_ssl,
                    "has_auth": bool(username and password),
                    "index_name": settings.opensearch.index_name,
                    "vector_size": settings.opensearch.vector_size
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Configuration", 
                False, 
                f"Failed to load configuration: {str(e)}"
            )
            return False
    
    def test_basic_connection(self):
        """Test basic OpenSearch connection."""
        try:
            host = settings.opensearch.host
            port = settings.opensearch.port
            use_ssl = settings.opensearch.use_ssl
            username = settings.opensearch.username
            password = settings.opensearch.password
            verify_certs = settings.opensearch.verify_certs
            
            # Build connection parameters
            client_params = {
                'hosts': [{'host': host, 'port': port}],
                'use_ssl': use_ssl,
                'verify_certs': verify_certs,
                'connection_class': RequestsHttpConnection,
                'timeout': 10,
                'max_retries': 1,
                'retry_on_timeout': False
            }
            
            # Add authentication if provided
            if username and password:
                client_params['http_auth'] = (username, password)
            
            # Additional SSL settings
            if use_ssl and not verify_certs:
                client_params['ssl_show_warn'] = False
                client_params['ssl_assert_hostname'] = False
            
            self.client = OpenSearch(**client_params)
            
            # Test connection with cluster health
            start_time = time.time()
            health = self.client.cluster.health()
            connection_time = time.time() - start_time
            
            self.log_result(
                "Basic Connection", 
                True, 
                f"Connected successfully",
                {
                    "connection_time_ms": round(connection_time * 1000, 2),
                    "cluster_name": health.get('cluster_name', 'unknown'),
                    "status": health.get('status', 'unknown'),
                    "number_of_nodes": health.get('number_of_nodes', 0)
                }
            )
            return True
            
        except ConnectionError as e:
            self.log_result(
                "Basic Connection", 
                False, 
                f"Connection failed: {str(e)}",
                {"error_type": "ConnectionError"}
            )
            return False
        except AuthenticationException as e:
            self.log_result(
                "Basic Connection", 
                False, 
                f"Authentication failed: {str(e)}",
                {"error_type": "AuthenticationException"}
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
    
    def test_cluster_info(self):
        """Test cluster information retrieval."""
        if not self.client:
            self.log_result("Cluster Info", False, "No client connection available")
            return False
        
        try:
            # Get cluster info
            info = self.client.info()
            stats = self.client.cluster.stats()
            
            self.log_result(
                "Cluster Info", 
                True, 
                f"Cluster information retrieved",
                {
                    "version": info.get('version', {}).get('number', 'unknown'),
                    "distribution": info.get('version', {}).get('distribution', 'unknown'),
                    "indices_count": stats.get('indices', {}).get('count', 0),
                    "docs_count": stats.get('indices', {}).get('docs', {}).get('count', 0)
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Cluster Info", 
                False, 
                f"Failed to get cluster info: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_index_operations(self):
        """Test index creation, deletion, and management."""
        if not self.client:
            self.log_result("Index Operations", False, "No client connection available")
            return False
        
        try:
            # Clean up any existing test index
            if self.client.indices.exists(index=self.test_index):
                self.client.indices.delete(index=self.test_index)
            
            # Create test index with vector mapping
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text"
                        },
                        "vector": {
                            "type": "knn_vector",
                            "dimension": settings.opensearch.vector_size,
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        "metadata": {
                            "type": "object"
                        },
                        "timestamp": {
                            "type": "date"
                        }
                    }
                }
            }
            
            # Create index
            create_response = self.client.indices.create(
                index=self.test_index,
                body=index_body
            )
            
            if not create_response.get('acknowledged'):
                self.log_result("Index Operations", False, "Index creation not acknowledged")
                return False
            
            # Wait for index to be ready
            time.sleep(1)
            
            # Check if index exists
            exists = self.client.indices.exists(index=self.test_index)
            
            if not exists:
                self.log_result("Index Operations", False, "Index does not exist after creation")
                return False
            
            # Get index info
            index_info = self.client.indices.get(index=self.test_index)
            
            self.log_result(
                "Index Operations", 
                True, 
                f"Index operations successful",
                {
                    "index_created": create_response.get('acknowledged', False),
                    "index_exists": exists,
                    "vector_dimension": settings.opensearch.vector_size,
                    "mappings_count": len(index_info.get(self.test_index, {}).get('mappings', {}).get('properties', {}))
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
    
    def test_document_operations(self):
        """Test document indexing, searching, and retrieval."""
        if not self.client:
            self.log_result("Document Operations", False, "No client connection available")
            return False
        
        try:
            # Create test documents
            test_docs = [
                {
                    "id": "test_doc_1",
                    "content": "This is a test document for OpenSearch connectivity testing",
                    "vector": [0.1] * settings.opensearch.vector_size,  # Simple test vector
                    "metadata": {
                        "test": True,
                        "doc_type": "connection_test"
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "id": "test_doc_2", 
                    "content": "Another test document with different content",
                    "vector": [0.2] * settings.opensearch.vector_size,  # Different test vector
                    "metadata": {
                        "test": True,
                        "doc_type": "connection_test"
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ]
            
            # Index documents
            indexed_count = 0
            for doc in test_docs:
                response = self.client.index(
                    index=self.test_index,
                    id=doc["id"],
                    body=doc
                )
                if response.get('result') in ['created', 'updated']:
                    indexed_count += 1
            
            # Refresh index to make documents searchable
            self.client.indices.refresh(index=self.test_index)
            
            # Wait for indexing to complete
            time.sleep(1)
            
            # Test document retrieval by ID
            retrieved_doc = self.client.get(
                index=self.test_index,
                id="test_doc_1"
            )
            
            if not retrieved_doc.get('found'):
                self.log_result("Document Operations", False, "Document retrieval by ID failed")
                return False
            
            # Test search
            search_body = {
                "query": {
                    "match": {
                        "content": "test document"
                    }
                },
                "size": 10
            }
            
            search_response = self.client.search(
                index=self.test_index,
                body=search_body
            )
            
            hits = search_response.get('hits', {}).get('hits', [])
            
            if len(hits) == 0:
                self.log_result("Document Operations", False, "Search returned no results")
                return False
            
            self.log_result(
                "Document Operations", 
                True, 
                f"Document operations successful",
                {
                    "documents_indexed": indexed_count,
                    "retrieval_success": retrieved_doc.get('found', False),
                    "search_hits": len(hits),
                    "total_hits": search_response.get('hits', {}).get('total', {}).get('value', 0)
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Document Operations", 
                False, 
                f"Document operations failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_vector_search(self):
        """Test vector similarity search functionality."""
        if not self.client:
            self.log_result("Vector Search", False, "No client connection available")
            return False
        
        try:
            # Test KNN vector search
            query_vector = [0.15] * settings.opensearch.vector_size  # Between our test vectors
            
            knn_body = {
                "size": 5,
                "query": {
                    "knn": {
                        "vector": {
                            "vector": query_vector,
                            "k": 2
                        }
                    }
                }
            }
            
            knn_response = self.client.search(
                index=self.test_index,
                body=knn_body
            )
            
            knn_hits = knn_response.get('hits', {}).get('hits', [])
            
            if len(knn_hits) == 0:
                self.log_result("Vector Search", False, "KNN search returned no results")
                return False
            
            # Check if results have scores
            has_scores = all('_score' in hit for hit in knn_hits)
            
            self.log_result(
                "Vector Search", 
                True, 
                f"Vector search successful",
                {
                    "knn_hits": len(knn_hits),
                    "has_scores": has_scores,
                    "vector_dimension": settings.opensearch.vector_size,
                    "top_score": knn_hits[0].get('_score', 0) if knn_hits else 0
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Vector Search", 
                False, 
                f"Vector search failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def cleanup(self):
        """Clean up test data and connections."""
        try:
            if self.client and self.client.indices.exists(index=self.test_index):
                self.client.indices.delete(index=self.test_index)
                
            self.log_result("Cleanup", True, "Test cleanup completed successfully")
            
        except Exception as e:
            self.log_result(
                "Cleanup", 
                False, 
                f"Cleanup failed: {str(e)}"
            )
    
    def run_all_tests(self):
        """Run all OpenSearch tests."""
        print("🔍 Starting OpenSearch Connection Tests")
        print("=" * 50)
        
        # Test sequence
        tests = [
            self.test_configuration,
            self.test_basic_connection,
            self.test_cluster_info,
            self.test_index_operations,
            self.test_document_operations,
            self.test_vector_search
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
            print("🎉 All OpenSearch tests passed!")
        else:
            print(f"⚠️  {total - passed} test(s) failed")
        
        # Always run cleanup
        self.cleanup()
        
        return passed == total


def main():
    """Main function to run OpenSearch connection tests."""
    
    print("🧪 OpenSearch Connection Test Suite")
    print("This script tests OpenSearch connectivity and vector operations")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/text_to_sql_rag"):
        print("❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run tests
    tester = OpenSearchConnectionTest()
    success = tester.run_all_tests()
    
    # Print configuration help
    print("\n" + "=" * 50)
    print("📝 Configuration Notes:")
    print(f"   OpenSearch Host: {settings.opensearch.host}")
    print(f"   OpenSearch Port: {settings.opensearch.port}")
    print(f"   Use SSL: {settings.opensearch.use_ssl}")
    print(f"   Index Name: {settings.opensearch.index_name}")
    print(f"   Vector Size: {settings.opensearch.vector_size}")
    print()
    print("   To configure OpenSearch connection:")
    print("   - Set OPENSEARCH_HOST environment variable")
    print("   - Set OPENSEARCH_PORT environment variable")
    print("   - Set OPENSEARCH_USE_SSL environment variable")
    print("   - Set OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD for auth")
    print("   - Or update src/text_to_sql_rag/config/settings.py")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())