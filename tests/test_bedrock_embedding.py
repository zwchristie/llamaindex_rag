#!/usr/bin/env python3
"""
AWS Bedrock Embedding Connection Test Script

Tests AWS Bedrock embedding model connectivity, authentication, and vector generation.
Can be run independently to validate Bedrock embedding service connectivity.
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import boto3
    from botocore.exceptions import (
        ClientError, NoCredentialsError, PartialCredentialsError,
        ProfileNotFound, EndpointConnectionError
    )
except ImportError as e:
    print(f"❌ ERROR: AWS SDK not installed: {e}")
    print("Install with: pip install boto3")
    sys.exit(1)

# Import our configuration and services
try:
    from text_to_sql_rag.config.settings import settings
    from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointEmbeddingService
except ImportError as e:
    print(f"❌ ERROR: Cannot import application modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class BedrockEmbeddingConnectionTest:
    """Test AWS Bedrock embedding connectivity and operations."""
    
    def __init__(self):
        self.bedrock_client = None
        self.bedrock_runtime_client = None
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
            "timestamp": datetime.utcnow().isoformat()
        }
        self.results.append(result)
        print(f"{status}: {test_name} - {message}")
        if details and not success:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def test_aws_credentials(self):
        """Test AWS credentials and configuration."""
        try:
            # Check credentials configuration
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                self.log_result(
                    "AWS Credentials", 
                    False, 
                    "No AWS credentials found"
                )
                return False
            
            # Test if credentials can be retrieved
            if not credentials.access_key:
                self.log_result(
                    "AWS Credentials", 
                    False, 
                    "AWS credentials incomplete - missing access key"
                )
                return False
            
            # Get region info
            region = session.region_name or settings.aws.region
            
            self.log_result(
                "AWS Credentials", 
                True, 
                f"AWS credentials configured",
                {
                    "region": region,
                    "has_access_key": bool(credentials.access_key),
                    "has_secret_key": bool(credentials.secret_key),
                    "has_session_token": bool(credentials.token)
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "AWS Credentials", 
                False, 
                f"Credentials error: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_bedrock_service_connection(self):
        """Test connection to AWS Bedrock service."""
        try:
            # Create Bedrock client
            self.bedrock_client = boto3.client(
                'bedrock',
                region_name=settings.aws.region
            )
            
            # Test connection by listing foundation models
            start_time = time.time()
            response = self.bedrock_client.list_foundation_models()
            connection_time = time.time() - start_time
            
            models = response.get('modelSummaries', [])
            
            # Find our target embedding model
            target_model = settings.aws.embedding_model
            available_models = [model.get('modelId') for model in models]
            target_model_available = target_model in available_models
            
            # Find embedding models
            embedding_models = [
                model.get('modelId') for model in models 
                if 'embed' in model.get('modelId', '').lower()
            ]
            
            self.log_result(
                "Bedrock Service Connection", 
                True, 
                f"Connected to Bedrock service",
                {
                    "connection_time_ms": round(connection_time * 1000, 2),
                    "total_models": len(models),
                    "target_model": target_model,
                    "target_model_available": target_model_available,
                    "embedding_models": embedding_models[:5]
                }
            )
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            self.log_result(
                "Bedrock Service Connection", 
                False, 
                f"AWS Client Error: {str(e)}",
                {"error_code": error_code, "error_type": "ClientError"}
            )
            return False
        except Exception as e:
            self.log_result(
                "Bedrock Service Connection", 
                False, 
                f"Unexpected error: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_bedrock_runtime_connection(self):
        """Test connection to Bedrock Runtime for embedding inference."""
        try:
            # Create Bedrock Runtime client
            self.bedrock_runtime_client = boto3.client(
                'bedrock-runtime',
                region_name=settings.aws.region
            )
            
            # Test with a simple embedding request
            model_id = settings.aws.embedding_model
            test_text = "This is a test for embedding generation"
            
            # Prepare request based on model type
            if 'amazon.titan-embed' in model_id:
                body = {
                    "inputText": test_text
                }
            elif 'cohere.embed' in model_id:
                body = {
                    "texts": [test_text],
                    "input_type": "search_document"
                }
            else:
                # Generic fallback
                body = {
                    "inputText": test_text
                }
            
            start_time = time.time()
            response = self.bedrock_runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            inference_time = time.time() - start_time
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            
            # Extract embedding based on model type
            if 'amazon.titan-embed' in model_id:
                embedding = response_body.get('embedding', [])
            elif 'cohere.embed' in model_id:
                embeddings = response_body.get('embeddings', [])
                embedding = embeddings[0] if embeddings else []
            else:
                embedding = response_body.get('embedding', [])
            
            if not embedding or len(embedding) == 0:
                self.log_result(
                    "Bedrock Runtime Connection", 
                    False, 
                    "Empty embedding response"
                )
                return False
            
            self.log_result(
                "Bedrock Runtime Connection", 
                True, 
                f"Runtime embedding inference successful",
                {
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model_id": model_id,
                    "embedding_dimension": len(embedding),
                    "embedding_preview": embedding[:5]  # First 5 values
                }
            )
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            self.log_result(
                "Bedrock Runtime Connection", 
                False, 
                f"Runtime Client Error: {str(e)}",
                {"error_code": error_code, "error_type": "ClientError"}
            )
            return False
        except Exception as e:
            self.log_result(
                "Bedrock Runtime Connection", 
                False, 
                f"Unexpected runtime error: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_embedding_generation(self):
        """Test embedding generation with various text inputs."""
        if not self.bedrock_runtime_client:
            self.log_result("Embedding Generation", False, "No Bedrock Runtime client available")
            return False
        
        try:
            model_id = settings.aws.embedding_model
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
                    
                    # Prepare request based on model type
                    if 'amazon.titan-embed' in model_id:
                        body = {
                            "inputText": text_content
                        }
                    elif 'cohere.embed' in model_id:
                        body = {
                            "texts": [text_content],
                            "input_type": "search_document"
                        }
                    else:
                        body = {
                            "inputText": text_content
                        }
                    
                    start_time = time.time()
                    response = self.bedrock_runtime_client.invoke_model(
                        modelId=model_id,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json"
                    )
                    generation_time = time.time() - start_time
                    
                    # Parse response
                    response_body = json.loads(response.get('body').read())
                    
                    # Extract embedding
                    if 'amazon.titan-embed' in model_id:
                        embedding = response_body.get('embedding', [])
                    elif 'cohere.embed' in model_id:
                        embeddings = response_body.get('embeddings', [])
                        embedding = embeddings[0] if embeddings else []
                    else:
                        embedding = response_body.get('embedding', [])
                    
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
                "Embedding Generation", 
                success, 
                f"Embedding generation tests: {successful_tests}/{total_tests} passed",
                {
                    "model_id": model_id,
                    "successful_texts": successful_tests,
                    "total_texts": total_tests,
                    "test_details": test_results
                }
            )
            return success
            
        except Exception as e:
            self.log_result(
                "Embedding Generation", 
                False, 
                f"Embedding generation test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def test_embedding_similarity(self):
        """Test embedding similarity calculations."""
        if not self.bedrock_runtime_client:
            self.log_result("Embedding Similarity", False, "No Bedrock Runtime client available")
            return False
        
        try:
            model_id = settings.aws.embedding_model
            
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
                if 'amazon.titan-embed' in model_id:
                    body = {"inputText": text}
                elif 'cohere.embed' in model_id:
                    body = {"texts": [text], "input_type": "search_document"}
                else:
                    body = {"inputText": text}
                
                response = self.bedrock_runtime_client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json"
                )
                
                response_body = json.loads(response.get('body').read())
                
                if 'amazon.titan-embed' in model_id:
                    embedding = response_body.get('embedding', [])
                elif 'cohere.embed' in model_id:
                    embedding_list = response_body.get('embeddings', [])
                    embedding = embedding_list[0] if embedding_list else []
                else:
                    embedding = response_body.get('embedding', [])
                
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
                "Embedding Similarity", 
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
                "Embedding Similarity", 
                False, 
                f"Embedding similarity test failed: {str(e)}",
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
    
    def test_vector_dimensions(self):
        """Test that embedding dimensions match configuration."""
        if not self.bedrock_runtime_client:
            self.log_result("Vector Dimensions", False, "No Bedrock Runtime client available")
            return False
        
        try:
            model_id = settings.aws.embedding_model
            expected_dimension = settings.opensearch.vector_size
            test_text = "Test vector dimension consistency"
            
            # Generate embedding
            if 'amazon.titan-embed' in model_id:
                body = {"inputText": test_text}
            elif 'cohere.embed' in model_id:
                body = {"texts": [test_text], "input_type": "search_document"}
            else:
                body = {"inputText": test_text}
            
            response = self.bedrock_runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            
            if 'amazon.titan-embed' in model_id:
                embedding = response_body.get('embedding', [])
            elif 'cohere.embed' in model_id:
                embedding_list = response_body.get('embeddings', [])
                embedding = embedding_list[0] if embedding_list else []
            else:
                embedding = response_body.get('embedding', [])
            
            actual_dimension = len(embedding)
            dimension_match = actual_dimension == expected_dimension
            
            self.log_result(
                "Vector Dimensions", 
                dimension_match, 
                f"Vector dimension check",
                {
                    "model_id": model_id,
                    "expected_dimension": expected_dimension,
                    "actual_dimension": actual_dimension,
                    "dimensions_match": dimension_match
                }
            )
            return dimension_match
            
        except Exception as e:
            self.log_result(
                "Vector Dimensions", 
                False, 
                f"Vector dimension test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def run_all_tests(self):
        """Run all Bedrock embedding tests."""
        print("🔍 Starting AWS Bedrock Embedding Connection Tests")
        print("=" * 50)
        
        # Test sequence
        tests = [
            self.test_aws_credentials,
            self.test_bedrock_service_connection,
            self.test_bedrock_runtime_connection,
            self.test_embedding_generation,
            self.test_embedding_similarity,
            self.test_vector_dimensions,
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
            print("🎉 All Bedrock embedding tests passed!")
        elif passed > 0:
            print(f"⚠️  {total - passed} test(s) failed, but some functionality is working")
        else:
            print("❌ All tests failed - check configuration and connectivity")
        
        return passed == total


def main():
    """Main function to run Bedrock embedding connection tests."""
    
    print("🧪 AWS Bedrock Embedding Connection Test Suite")
    print("This script tests AWS Bedrock embedding connectivity and vector generation")
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
    print(f"   AWS Region: {settings.aws.region}")
    print(f"   Embedding Model: {settings.aws.embedding_model}")
    print(f"   Expected Vector Size: {settings.opensearch.vector_size}")
    print(f"   Bedrock Endpoint: {settings.bedrock_endpoint_url or 'Not configured'}")
    print()
    print("   To configure AWS Bedrock Embeddings:")
    print("   - Set AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("   - Set AWS_REGION environment variable")
    print("   - Set AWS_EMBEDDING_MODEL for specific model")
    print("   - Set OPENSEARCH_VECTOR_SIZE to match model dimensions")
    print("   - Set BEDROCK_ENDPOINT_URL for custom endpoints")
    print("   - Ensure IAM permissions for bedrock:InvokeModel")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())