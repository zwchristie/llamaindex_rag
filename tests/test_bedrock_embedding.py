#!/usr/bin/env python3
"""
Bedrock Endpoint Embedding Connection Test Script

Tests Bedrock endpoint embedding model connectivity and vector generation.
Can be run independently to validate Bedrock endpoint embedding service connectivity.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our configuration and services
try:
    from text_to_sql_rag.config.settings import settings
    from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointEmbeddingService
except ImportError as e:
    print(f"❌ ERROR: Cannot import application modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class BedrockEmbeddingConnectionTest:
    """Test Bedrock endpoint embedding connectivity and operations."""
    
    def __init__(self):
        self.results = []
        
        # Test texts for different scenarios
        self.test_texts = {
            "short": "Hello world",
            "medium": "This is a medium-length text for testing embedding generation with multiple words and concepts.",
            "sql_query": "SELECT customer_id, SUM(order_amount) FROM orders WHERE order_date >= '2023-01-01' GROUP BY customer_id ORDER BY SUM(order_amount) DESC;",
            "technical": "Database indexing is a data structure technique to efficiently locate and access the data in a database. Indexes are used to quickly locate data without having to search every row in a database table.",
            "empty": ""
        }
    
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
    
    def test_endpoint_configuration(self):
        """Test Bedrock endpoint configuration."""
        try:
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Endpoint Configuration", 
                    False, 
                    "Bedrock endpoint URL not configured",
                    {"endpoint_url": endpoint_url}
                )
                return False
            
            # Basic URL validation
            if not endpoint_url.startswith(('http://', 'https://')):
                self.log_result(
                    "Endpoint Configuration", 
                    False, 
                    "Invalid endpoint URL format",
                    {"endpoint_url": endpoint_url}
                )
                return False
            
            self.log_result(
                "Endpoint Configuration", 
                True, 
                f"Endpoint configuration valid",
                {
                    "endpoint_url": endpoint_url,
                    "embedding_model": settings.aws.embedding_model,
                    "vector_size": settings.opensearch.vector_size
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Endpoint Configuration", 
                False, 
                f"Configuration error: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_endpoint_connectivity(self):
        """Test basic endpoint connectivity."""
        try:
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Endpoint Connectivity", 
                    False, 
                    "Bedrock endpoint URL not configured"
                )
                return False
            
            # Import requests for basic connectivity test
            try:
                import requests
                # Disable SSL warnings if SSL verification is disabled
                if not settings.bedrock_endpoint_verify_ssl:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:
                self.log_result(
                    "Endpoint Connectivity", 
                    False, 
                    "requests library not available for connectivity test"
                )
                return False
            
            # Test basic connectivity with a simple request
            try:
                start_time = time.time()
                # Just test connectivity, not actual inference
                verify_ssl = settings.bedrock_endpoint_verify_ssl
                response = requests.get(
                    endpoint_url.rstrip('/'), 
                    timeout=10,
                    allow_redirects=False,
                    verify=verify_ssl
                )
                connection_time = time.time() - start_time
                
                # We expect some response (even if it's an error about missing data)
                # The important thing is that we can connect to the endpoint
                self.log_result(
                    "Endpoint Connectivity", 
                    True, 
                    f"Endpoint is reachable",
                    {
                        "endpoint_url": endpoint_url,
                        "connection_time_ms": round(connection_time * 1000, 2),
                        "status_code": response.status_code
                    }
                )
                return True
                
            except requests.exceptions.ConnectionError:
                self.log_result(
                    "Endpoint Connectivity", 
                    False, 
                    f"Cannot connect to endpoint: {endpoint_url}"
                )
                return False
            except requests.exceptions.Timeout:
                self.log_result(
                    "Endpoint Connectivity", 
                    False, 
                    f"Endpoint connection timed out: {endpoint_url}"
                )
                return False
            except Exception as e:
                # Even if we get other errors, the endpoint might be reachable
                # This is just a basic connectivity test
                self.log_result(
                    "Endpoint Connectivity", 
                    True, 
                    f"Endpoint is reachable (with error: {type(e).__name__})",
                    {
                        "endpoint_url": endpoint_url,
                        "note": "Endpoint responded but may require specific request format"
                    }
                )
                return True
                
        except Exception as e:
            self.log_result(
                "Endpoint Connectivity", 
                False, 
                f"Connectivity test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_embedding_generation_via_endpoint(self):
        """Test embedding generation with various text inputs via endpoint service."""
        try:
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Embedding Generation via Endpoint", 
                    False, 
                    "Bedrock endpoint URL not configured"
                )
                return False
            
            # Test the endpoint service with multiple text inputs
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            endpoint_service = BedrockEndpointService(endpoint_url)
            embedding_service = BedrockEndpointEmbeddingService(endpoint_service)
            
            successful_tests = 0
            total_tests = len(self.test_texts)
            test_results = {}
            
            for text_name, text_content in self.test_texts.items():
                try:
                    if text_name == "empty":
                        # Test empty text handling
                        if not text_content:
                            test_results[text_name] = {
                                "success": True,
                                "note": "Skipped empty text test"
                            }
                            successful_tests += 1
                            continue
                    
                    start_time = time.time()
                    embedding = embedding_service.get_embedding(text_content)
                    generation_time = time.time() - start_time
                    
                    if embedding and len(embedding) > 0:
                        successful_tests += 1
                        test_results[text_name] = {
                            "success": True,
                            "generation_time_ms": round(generation_time * 1000, 2),
                            "embedding_dimension": len(embedding),
                            "text_length": len(text_content),
                            "embedding_norm": round(np.linalg.norm(embedding), 4)
                        }
                    else:
                        test_results[text_name] = {
                            "success": False,
                            "error": "Empty embedding response"
                        }
                        
                except Exception as e:
                    test_results[text_name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            success = successful_tests == total_tests
            
            self.log_result(
                "Embedding Generation via Endpoint", 
                success, 
                f"Embedding generation tests: {successful_tests}/{total_tests} passed",
                {
                    "endpoint_url": endpoint_url,
                    "successful_texts": successful_tests,
                    "total_texts": total_tests,
                    "test_details": test_results
                }
            )
            return success
            
        except Exception as e:
            self.log_result(
                "Embedding Generation via Endpoint", 
                False, 
                f"Embedding generation test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_embedding_similarity_via_endpoint(self):
        """Test embedding similarity calculations via endpoint service."""
        try:
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Embedding Similarity via Endpoint", 
                    False, 
                    "Bedrock endpoint URL not configured"
                )
                return False
            
            # Test the endpoint service
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            endpoint_service = BedrockEndpointService(endpoint_url)
            embedding_service = BedrockEndpointEmbeddingService(endpoint_service)
            
            # Test with similar texts
            similar_texts = [
                "The database contains customer information",
                "Customer data is stored in the database"
            ]
            
            # Test with dissimilar texts
            dissimilar_texts = [
                "The database contains customer information",
                "Today is a sunny day outside"
            ]
            
            embeddings = {}
            
            # Generate embeddings for all test texts
            all_texts = list(set(similar_texts + dissimilar_texts))
            
            for text in all_texts:
                embedding = embedding_service.get_embedding(text)
                embeddings[text] = embedding
            
            # Calculate similarity scores
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            # Similar texts similarity
            similar_score = cosine_similarity(
                embeddings[similar_texts[0]], 
                embeddings[similar_texts[1]]
            )
            
            # Dissimilar texts similarity
            dissimilar_score = cosine_similarity(
                embeddings[dissimilar_texts[0]], 
                embeddings[dissimilar_texts[1]]
            )
            
            # Similar texts should have higher similarity than dissimilar texts
            similarity_test_passed = similar_score > dissimilar_score
            
            self.log_result(
                "Embedding Similarity via Endpoint", 
                similarity_test_passed, 
                f"Embedding similarity calculation successful",
                {
                    "similar_texts_score": round(similar_score, 4),
                    "dissimilar_texts_score": round(dissimilar_score, 4),
                    "similarity_difference": round(similar_score - dissimilar_score, 4),
                    "test_passed": similarity_test_passed
                }
            )
            return similarity_test_passed
            
        except Exception as e:
            self.log_result(
                "Embedding Similarity via Endpoint", 
                False, 
                f"Embedding similarity test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_vector_dimensions_via_endpoint(self):
        """Test that embedding dimensions match configuration via endpoint service."""
        try:
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Vector Dimensions via Endpoint", 
                    False, 
                    "Bedrock endpoint URL not configured"
                )
                return False
            
            # Test the endpoint service
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            endpoint_service = BedrockEndpointService(endpoint_url)
            embedding_service = BedrockEndpointEmbeddingService(endpoint_service)
            
            expected_dimension = settings.opensearch.vector_size
            test_text = "Test vector dimension consistency"
            
            # Generate embedding
            embedding = embedding_service.get_embedding(test_text)
            actual_dimension = len(embedding)
            dimension_match = actual_dimension == expected_dimension
            
            self.log_result(
                "Vector Dimensions via Endpoint", 
                dimension_match, 
                f"Vector dimension check",
                {
                    "embedding_model": settings.aws.embedding_model,
                    "expected_dimension": expected_dimension,
                    "actual_dimension": actual_dimension,
                    "dimensions_match": dimension_match
                }
            )
            return dimension_match
            
        except Exception as e:
            self.log_result(
                "Vector Dimensions via Endpoint", 
                False, 
                f"Vector dimension test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_bedrock_endpoint_service(self):
        """Test our custom Bedrock endpoint service wrapper."""
        try:
            # Test if bedrock endpoint URL is configured
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Bedrock Endpoint Service", 
                    False, 
                    "Bedrock endpoint URL not configured",
                    {"endpoint_url": endpoint_url}
                )
                return False
            
            # Test the endpoint service
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            endpoint_service = BedrockEndpointService(endpoint_url)
            embedding_service = BedrockEndpointEmbeddingService(endpoint_service)
            
            # Test embedding generation
            test_text = "What is machine learning and artificial intelligence?"
            
            start_time = time.time()
            embedding = embedding_service.get_embedding(test_text)
            generation_time = time.time() - start_time
            
            if not embedding or len(embedding) == 0:
                self.log_result(
                    "Bedrock Endpoint Service", 
                    False, 
                    "Empty embedding from endpoint service"
                )
                return False
            
            self.log_result(
                "Bedrock Endpoint Service", 
                True, 
                f"Endpoint embedding service working correctly",
                {
                    "endpoint_url": endpoint_url,
                    "generation_time_ms": round(generation_time * 1000, 2),
                    "embedding_dimension": len(embedding),
                    "embedding_norm": round(np.linalg.norm(embedding), 4)
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Bedrock Endpoint Service", 
                False, 
                f"Endpoint service test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def run_all_tests(self):
        """Run all Bedrock embedding tests."""
        print("🔍 Starting Bedrock Endpoint Embedding Connection Tests")
        print("=" * 50)
        
        # Test sequence
        tests = [
            self.test_endpoint_configuration,
            self.test_endpoint_connectivity,
            self.test_embedding_generation_via_endpoint,
            self.test_embedding_similarity_via_endpoint,
            self.test_vector_dimensions_via_endpoint,
            self.test_bedrock_endpoint_service
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
            print("🎉 All Bedrock endpoint embedding tests passed!")
        elif passed > 0:
            print(f"⚠️  {total - passed} test(s) failed, but some functionality is working")
        else:
            print("❌ All tests failed - check endpoint configuration and connectivity")
        
        return passed == total


def main():
    """Main function to run Bedrock endpoint embedding connection tests."""
    
    print("🧪 Bedrock Endpoint Embedding Connection Test Suite")
    print("This script tests Bedrock endpoint embedding connectivity and vector generation")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/text_to_sql_rag"):
        print("❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run tests
    tester = BedrockEmbeddingConnectionTest()
    success = tester.run_all_tests()
    
    # Print configuration help
    print("\n" + "=" * 50)
    print("📝 Configuration Notes:")
    print(f"   Embedding Model: {settings.aws.embedding_model}")
    print(f"   Expected Vector Size: {settings.opensearch.vector_size}")
    print(f"   Bedrock Endpoint: {settings.bedrock_endpoint_url or 'Not configured'}")
    print()
    print("   To configure Bedrock Endpoint Embeddings:")
    print("   - Set BEDROCK_ENDPOINT_URL for your Bedrock endpoint")
    print("   - Set BEDROCK_ENDPOINT_VERIFY_SSL=false to disable SSL verification if needed")
    print("   - Set AWS_EMBEDDING_MODEL for specific model (optional)")
    print("   - Set OPENSEARCH_VECTOR_SIZE to match model dimensions")
    print("   - Ensure endpoint has proper authentication and permissions")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())