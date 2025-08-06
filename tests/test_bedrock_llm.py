#!/usr/bin/env python3
"""
AWS Bedrock LLM Connection Test Script

Tests AWS Bedrock LLM connectivity, authentication, and text generation.
Can be run independently to validate Bedrock LLM service connectivity.
"""

import os
import sys
import time
import json
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
    from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointLLMWrapper
    from text_to_sql_rag.services.llm_provider_factory import llm_factory
except ImportError as e:
    print(f"❌ ERROR: Cannot import application modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class BedrockLLMConnectionTest:
    """Test AWS Bedrock LLM connectivity and operations."""
    
    def __init__(self):
        self.bedrock_client = None
        self.bedrock_runtime_client = None
        self.results = []
        
        # Test prompts for different scenarios
        self.test_prompts = {
            "simple": "What is 2 + 2?",
            "sql": "Generate a simple SQL query to select all columns from a table named 'users'.",
            "explanation": "Explain what a database index is and why it's useful.",
            "complex": "You are a SQL expert. Convert this natural language query to SQL: Show me the top 5 customers by total order amount in 2023."
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
            
        except NoCredentialsError:
            self.log_result(
                "AWS Credentials", 
                False, 
                "AWS credentials not found",
                {"error_type": "NoCredentialsError"}
            )
            return False
        except PartialCredentialsError as e:
            self.log_result(
                "AWS Credentials", 
                False, 
                f"AWS credentials incomplete: {str(e)}",
                {"error_type": "PartialCredentialsError"}
            )
            return False
        except ProfileNotFound as e:
            self.log_result(
                "AWS Credentials", 
                False, 
                f"AWS profile not found: {str(e)}",
                {"error_type": "ProfileNotFound"}
            )
            return False
        except Exception as e:
            self.log_result(
                "AWS Credentials", 
                False, 
                f"Unexpected credentials error: {str(e)}",
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
            
            # Find our target model
            target_model = settings.aws.llm_model
            available_models = [model.get('modelId') for model in models]
            target_model_available = target_model in available_models
            
            self.log_result(
                "Bedrock Service Connection", 
                True, 
                f"Connected to Bedrock service",
                {
                    "connection_time_ms": round(connection_time * 1000, 2),
                    "total_models": len(models),
                    "target_model": target_model,
                    "target_model_available": target_model_available,
                    "sample_models": available_models[:5]
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
        except EndpointConnectionError as e:
            self.log_result(
                "Bedrock Service Connection", 
                False, 
                f"Endpoint connection failed: {str(e)}",
                {"error_type": "EndpointConnectionError"}
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
        """Test connection to Bedrock Runtime for inference."""
        try:
            # Create Bedrock Runtime client
            self.bedrock_runtime_client = boto3.client(
                'bedrock-runtime',
                region_name=settings.aws.region
            )
            
            # Test with a simple inference request
            model_id = settings.aws.llm_model
            
            # Prepare request based on model type
            if 'anthropic.claude' in model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": "Say 'Hello from Bedrock'"}
                    ]
                }
            else:
                # Fallback for other models
                body = {
                    "inputText": "Say 'Hello from Bedrock'",
                    "textGenerationConfig": {
                        "maxTokenCount": 100,
                        "temperature": 0.1
                    }
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
            
            # Extract generated text based on model type
            if 'anthropic.claude' in model_id:
                generated_text = response_body.get('content', [{}])[0].get('text', '')
            else:
                generated_text = response_body.get('results', [{}])[0].get('outputText', '')
            
            self.log_result(
                "Bedrock Runtime Connection", 
                True, 
                f"Runtime inference successful",
                {
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "model_id": model_id,
                    "response_length": len(generated_text),
                    "generated_preview": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text
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
    
    def test_text_generation(self):
        """Test text generation with various prompts."""
        if not self.bedrock_runtime_client:
            self.log_result("Text Generation", False, "No Bedrock Runtime client available")
            return False
        
        try:
            model_id = settings.aws.llm_model
            successful_tests = 0
            total_tests = len(self.test_prompts)
            
            test_results = {}
            
            for prompt_name, prompt_text in self.test_prompts.items():
                try:
                    # Prepare request
                    if 'anthropic.claude' in model_id:
                        body = {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 200,
                            "temperature": 0.1,
                            "messages": [
                                {"role": "user", "content": prompt_text}
                            ]
                        }
                    else:
                        body = {
                            "inputText": prompt_text,
                            "textGenerationConfig": {
                                "maxTokenCount": 200,
                                "temperature": 0.1
                            }
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
                    
                    # Extract generated text
                    if 'anthropic.claude' in model_id:
                        generated_text = response_body.get('content', [{}])[0].get('text', '')
                    else:
                        generated_text = response_body.get('results', [{}])[0].get('outputText', '')
                    
                    if generated_text and len(generated_text) > 0:
                        successful_tests += 1
                        test_results[prompt_name] = {
                            "success": True,
                            "generation_time_ms": round(generation_time * 1000, 2),
                            "response_length": len(generated_text),
                            "preview": generated_text[:50] + "..." if len(generated_text) > 50 else generated_text
                        }
                    else:
                        test_results[prompt_name] = {
                            "success": False,
                            "error": "Empty response"
                        }
                        
                except Exception as e:
                    test_results[prompt_name] = {
                        "success": False,
                        "error": str(e)
                    }
            
            success = successful_tests == total_tests
            
            self.log_result(
                "Text Generation", 
                success, 
                f"Text generation tests: {successful_tests}/{total_tests} passed",
                {
                    "model_id": model_id,
                    "successful_prompts": successful_tests,
                    "total_prompts": total_tests,
                    "test_details": test_results
                }
            )
            return success
            
        except Exception as e:
            self.log_result(
                "Text Generation", 
                False, 
                f"Text generation test failed: {str(e)}",
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
            llm_wrapper = BedrockEndpointLLMWrapper(endpoint_service)
            
            # Test text generation
            test_prompt = "What is machine learning?"
            
            start_time = time.time()
            response = llm_wrapper.generate_response(test_prompt)
            generation_time = time.time() - start_time
            
            if not response or len(response.strip()) == 0:
                self.log_result(
                    "Bedrock Endpoint Service", 
                    False, 
                    "Empty response from endpoint service"
                )
                return False
            
            self.log_result(
                "Bedrock Endpoint Service", 
                True, 
                f"Endpoint service working correctly",
                {
                    "endpoint_url": endpoint_url,
                    "generation_time_ms": round(generation_time * 1000, 2),
                    "response_length": len(response),
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
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
    
    def test_llm_factory_service(self):
        """Test our LLM factory service."""
        try:
            # Test provider info
            provider_info = llm_factory.get_provider_info()
            
            # Test health check
            health_status = llm_factory.health_check()
            
            # Test text generation through factory
            test_prompt = "Explain what SQL is in one sentence."
            
            start_time = time.time()
            response = llm_factory.generate_text(test_prompt)
            generation_time = time.time() - start_time
            
            if not response or len(response.strip()) == 0:
                self.log_result(
                    "LLM Factory Service", 
                    False, 
                    "Empty response from LLM factory"
                )
                return False
            
            self.log_result(
                "LLM Factory Service", 
                True, 
                f"LLM factory service working correctly",
                {
                    "provider": provider_info.get("provider", "unknown"),
                    "health_status": health_status,
                    "generation_time_ms": round(generation_time * 1000, 2),
                    "response_length": len(response),
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                }
            )
            return True
            
        except Exception as e:
            self.log_result(
                "LLM Factory Service", 
                False, 
                f"LLM factory test failed: {str(e)}",
                {"error_type": type(e).__name__}
            )
            return False
    
    def run_all_tests(self):
        """Run all Bedrock LLM tests."""
        print("🔍 Starting AWS Bedrock LLM Connection Tests")
        print("=" * 50)
        
        # Test sequence
        tests = [
            self.test_aws_credentials,
            self.test_bedrock_service_connection,
            self.test_bedrock_runtime_connection,
            self.test_text_generation,
            self.test_bedrock_endpoint_service,
            self.test_llm_factory_service
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
            print("🎉 All Bedrock LLM tests passed!")
        elif passed > 0:
            print(f"⚠️  {total - passed} test(s) failed, but some functionality is working")
        else:
            print("❌ All tests failed - check configuration and connectivity")
        
        return passed == total


def main():
    """Main function to run Bedrock LLM connection tests."""
    
    print("🧪 AWS Bedrock LLM Connection Test Suite")
    print("This script tests AWS Bedrock LLM connectivity and text generation")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("src/text_to_sql_rag"):
        print("❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Run tests
    tester = BedrockLLMConnectionTest()
    success = tester.run_all_tests()
    
    # Print configuration help
    print("\n" + "=" * 50)
    print("📝 Configuration Notes:")
    print(f"   AWS Region: {settings.aws.region}")
    print(f"   LLM Model: {settings.aws.llm_model}")
    print(f"   Bedrock Endpoint: {settings.bedrock_endpoint_url or 'Not configured'}")
    print(f"   LLM Provider: {settings.llm_provider.provider}")
    print()
    print("   To configure AWS Bedrock:")
    print("   - Set AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
    print("   - Set AWS_REGION environment variable")
    print("   - Set AWS_LLM_MODEL for specific model")
    print("   - Set BEDROCK_ENDPOINT_URL for custom endpoints")
    print("   - Ensure IAM permissions for bedrock:InvokeModel")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())