#!/usr/bin/env python3
"""
Test Bedrock LLM functionality using actual application services.
Tests the real connection and functionality, not separate test connections.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BedrockLLMTest:
    """Test Bedrock LLM using actual application services."""
    
    def __init__(self):
        self.llm_factory = None
        self.results = []
    
    def log_result(self, test_name: str, success: bool, message: str, details: dict = None):
        """Log test result."""
        status = "PASS" if success else "FAIL"
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
            import os
            # Set required environment variables if not already set
            if not os.getenv('BEDROCK_ENDPOINT_URL'):
                os.environ['BEDROCK_ENDPOINT_URL'] = 'https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess'
            if not os.getenv('AWS_LLM_MODEL'):
                os.environ['AWS_LLM_MODEL'] = 'anthropic.claude-3-haiku-20240307-v1:0'
            if not os.getenv('AWS_EMBEDDING_MODEL'):
                os.environ['AWS_EMBEDDING_MODEL'] = 'amazon.titan-embed-text-v2:0'
            if not os.getenv('LLM_PROVIDER'):
                os.environ['LLM_PROVIDER'] = 'bedrock_endpoint'
            
            # Import settings and patch the global settings instance
            from text_to_sql_rag.config import settings as settings_module
            from text_to_sql_rag.services.enhanced_bedrock_service import EnhancedBedrockService, EnhancedBedrockLLMWrapper
            
            # Patch the global settings to use environment variables
            if not settings_module.settings.bedrock_endpoint.url and os.getenv('BEDROCK_ENDPOINT_URL'):
                settings_module.settings.bedrock_endpoint.url = os.getenv('BEDROCK_ENDPOINT_URL')
            
            self.settings = settings_module.settings
            
            # Now create a new LLM factory with the updated settings
            from text_to_sql_rag.services.llm_provider_factory import LLMProviderFactory
            self.llm_factory = LLMProviderFactory()
            
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
        """Test LLM provider configuration."""
        try:
            import os
            
            # Check configuration using the fresh settings instance
            bedrock_url = self.settings.bedrock_endpoint.url
            provider_type = self.settings.llm_provider.provider
            
            # Debug environment variable
            env_bedrock_url = os.getenv('BEDROCK_ENDPOINT_URL')
            
            if not bedrock_url:
                self.log_result(
                    "Configuration",
                    False,
                    "BEDROCK_ENDPOINT_URL not configured",
                    {
                        "settings_bedrock_url": bedrock_url,
                        "env_bedrock_url": env_bedrock_url,
                        "provider": provider_type
                    }
                )
                return False
            
            self.log_result(
                "Configuration",
                True,
                f"Configuration loaded successfully",
                {
                    "bedrock_url": bedrock_url[:50] + "..." if len(bedrock_url) > 50 else bedrock_url,
                    "provider": provider_type,
                    "verify_ssl": self.settings.bedrock_endpoint.verify_ssl
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
    
    def test_provider_initialization(self):
        """Test LLM provider initialization."""
        try:
            provider_info = self.llm_factory.get_provider_info()
            
            self.log_result(
                "Provider Initialization",
                True,
                f"Provider initialized successfully",
                provider_info
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Provider Initialization",
                False,
                f"Provider initialization failed: {e}"
            )
            return False
    
    def test_configuration_check(self):
        """Test LLM provider configuration check (no API calls)."""
        try:
            is_configured = self.llm_factory.is_configured()
            
            if is_configured:
                self.log_result(
                    "Configuration Check",
                    True,
                    "LLM provider is properly configured"
                )
            else:
                self.log_result(
                    "Configuration Check",
                    False,
                    "LLM provider configuration check failed"
                )
            
            return is_configured
            
        except Exception as e:
            self.log_result(
                "Configuration Check",
                False,
                f"Configuration check error: {e}"
            )
            return False
    
    def test_text_generation(self):
        """Test basic text generation."""
        try:
            test_prompt = "Generate a simple greeting message."
            
            response = self.llm_factory.generate_text(test_prompt)
            
            if response and len(response.strip()) > 0:
                self.log_result(
                    "Text Generation",
                    True,
                    f"Successfully generated text",
                    {
                        "prompt": test_prompt,
                        "response_length": len(response),
                        "response_preview": response[:100] + "..." if len(response) > 100 else response
                    }
                )
                return True
            else:
                self.log_result(
                    "Text Generation",
                    False,
                    "Generated text is empty"
                )
                return False
            
        except Exception as e:
            self.log_result(
                "Text Generation",
                False,
                f"Text generation failed: {e}"
            )
            return False
    
    def test_sql_generation(self):
        """Test SQL query generation."""
        try:
            natural_query = "Show me all users who logged in yesterday"
            schema_context = """
            Table: users
            Columns: user_id (NUMBER), username (VARCHAR2), email (VARCHAR2), last_login (DATE), status (VARCHAR2)
            """
            
            result = self.llm_factory.generate_sql_query(
                natural_language_query=natural_query,
                schema_context=schema_context
            )
            
            if result and "sql" in result:
                self.log_result(
                    "SQL Generation",
                    True,
                    f"Successfully generated SQL",
                    {
                        "natural_query": natural_query,
                        "sql_preview": result["sql"][:100] + "..." if len(result["sql"]) > 100 else result["sql"],
                        "has_explanation": bool(result.get("explanation"))
                    }
                )
                return True
            else:
                self.log_result(
                    "SQL Generation",
                    False,
                    "SQL generation returned empty result",
                    {"result": result}
                )
                return False
            
        except Exception as e:
            self.log_result(
                "SQL Generation",
                False,
                f"SQL generation failed: {e}"
            )
            return False
    
    def test_connection_resilience(self):
        """Test connection resilience with multiple requests."""
        try:
            success_count = 0
            total_requests = 3
            
            for i in range(total_requests):
                try:
                    response = self.llm_factory.generate_text(f"Test request {i+1}")
                    if response and len(response.strip()) > 0:
                        success_count += 1
                except Exception as e:
                    logger.warning(f"Request {i+1} failed: {e}")
            
            success_rate = success_count / total_requests
            
            if success_rate >= 0.8:  # 80% success rate
                self.log_result(
                    "Connection Resilience",
                    True,
                    f"Connection resilience test passed",
                    {
                        "success_rate": f"{success_rate:.2%}",
                        "successful_requests": success_count,
                        "total_requests": total_requests
                    }
                )
                return True
            else:
                self.log_result(
                    "Connection Resilience",
                    False,
                    f"Connection resilience test failed",
                    {
                        "success_rate": f"{success_rate:.2%}",
                        "successful_requests": success_count,
                        "total_requests": total_requests
                    }
                )
                return False
            
        except Exception as e:
            self.log_result(
                "Connection Resilience",
                False,
                f"Connection resilience test error: {e}"
            )
            return False
    
    async def run_all_tests(self):
        """Run all Bedrock LLM tests."""
        print("Starting Bedrock LLM Tests")
        print("=" * 50)
        
        # Sync tests first
        tests_sync = [
            self.test_import_services,
            self.test_configuration,
            self.test_provider_initialization,
            self.test_configuration_check
        ]
        
        # Sync tests (after provider initialization)
        tests_sync_after = [
            self.test_text_generation,
            self.test_sql_generation,
            self.test_connection_resilience
        ]
        
        # Async tests (none currently)
        tests_async = []
        
        passed = 0
        total = len(tests_sync) + len(tests_async) + len(tests_sync_after)
        
        # Run sync tests
        for test in tests_sync:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"FAIL: {test.__name__} - Unexpected error: {e}")
        
        # Run async tests
        for test in tests_async:
            try:
                if await test():
                    passed += 1
            except Exception as e:
                print(f"FAIL: {test.__name__} - Unexpected error: {e}")
        
        # Run sync tests after provider initialization
        for test in tests_sync_after:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"FAIL: {test.__name__} - Unexpected error: {e}")
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("All Bedrock LLM tests passed!")
        else:
            print(f"{total - passed} test(s) failed")
        
        return passed == total


async def main():
    """Main function to run Bedrock LLM tests."""
    print("Bedrock LLM Test Suite")
    print("This tests the actual Bedrock LLM functionality using application services")
    print()
    
    tester = BedrockLLMTest()
    success = await tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)