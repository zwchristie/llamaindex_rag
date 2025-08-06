#!/usr/bin/env python3
"""
Bedrock Endpoint LLM Connection Test Script

Tests Bedrock endpoint LLM connectivity and text generation.
Can be run independently to validate Bedrock endpoint service connectivity.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add src to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    """Test Bedrock endpoint LLM connectivity and operations."""
    
    def __init__(self):
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
                    "llm_model": settings.aws.llm_model,
                    "embedding_model": settings.aws.embedding_model
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
                response = requests.get(
                    endpoint_url.rstrip('/'), 
                    timeout=10,
                    allow_redirects=False
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
    
    def test_text_generation_via_endpoint(self):
        """Test text generation with various prompts via endpoint service."""
        try:
            endpoint_url = settings.bedrock_endpoint_url
            
            if not endpoint_url:
                self.log_result(
                    "Text Generation via Endpoint", 
                    False, 
                    "Bedrock endpoint URL not configured"
                )
                return False
            
            # Test the endpoint service with multiple prompts
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            endpoint_service = BedrockEndpointService(endpoint_url)
            llm_wrapper = BedrockEndpointLLMWrapper(endpoint_service)
            
            successful_tests = 0
            total_tests = len(self.test_prompts)
            test_results = {}
            
            for prompt_name, prompt_text in self.test_prompts.items():
                try:
                    start_time = time.time()
                    response = llm_wrapper.generate_response(prompt_text)
                    generation_time = time.time() - start_time
                    
                    if response and len(response.strip()) > 0:
                        successful_tests += 1
                        test_results[prompt_name] = {
                            "success": True,
                            "generation_time_ms": round(generation_time * 1000, 2),
                            "response_length": len(response),
                            "preview": response[:50] + "..." if len(response) > 50 else response
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
                "Text Generation via Endpoint", 
                success, 
                f"Text generation tests: {successful_tests}/{total_tests} passed",
                {
                    "endpoint_url": endpoint_url,
                    "successful_prompts": successful_tests,
                    "total_prompts": total_tests,
                    "test_details": test_results
                }
            )
            return success
            
        except Exception as e:
            self.log_result(
                "Text Generation via Endpoint", 
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
        print("🔍 Starting Bedrock Endpoint LLM Connection Tests")
        print("=" * 50)
        
        # Test sequence
        tests = [
            self.test_endpoint_configuration,
            self.test_endpoint_connectivity,
            self.test_text_generation_via_endpoint,
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
            print("🎉 All Bedrock endpoint LLM tests passed!")
        elif passed > 0:
            print(f"⚠️  {total - passed} test(s) failed, but some functionality is working")
        else:
            print("❌ All tests failed - check endpoint configuration and connectivity")
        
        return passed == total


def main():
    """Main function to run Bedrock endpoint LLM connection tests."""
    
    print("🧪 Bedrock Endpoint LLM Connection Test Suite")
    print("This script tests Bedrock endpoint LLM connectivity and text generation")
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
    print(f"   LLM Model: {settings.aws.llm_model}")
    print(f"   Embedding Model: {settings.aws.embedding_model}")
    print(f"   Bedrock Endpoint: {settings.bedrock_endpoint_url or 'Not configured'}")
    print(f"   LLM Provider: {settings.llm_provider.provider}")
    print()
    print("   To configure Bedrock Endpoint:")
    print("   - Set BEDROCK_ENDPOINT_URL for your Bedrock endpoint")
    print("   - Set AWS_LLM_MODEL for specific model (optional)")
    print("   - Set AWS_EMBEDDING_MODEL for specific embedding model (optional)")
    print("   - Set LLM_PROVIDER=bedrock")
    print("   - Ensure endpoint has proper authentication and permissions")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())