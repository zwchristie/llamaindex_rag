#!/usr/bin/env python3
"""
Test Bedrock embedding functionality using actual application services.
Tests the real connection and functionality, not separate test connections.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BedrockEmbeddingTest:
    """Test Bedrock embedding using actual application services."""
    
    def __init__(self):
        self.bedrock_service = None
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
            from text_to_sql_rag.services.enhanced_bedrock_service import EnhancedBedrockService
            
            # Patch the global settings to use environment variables
            if not settings_module.settings.bedrock_endpoint.url and os.getenv('BEDROCK_ENDPOINT_URL'):
                settings_module.settings.bedrock_endpoint.url = os.getenv('BEDROCK_ENDPOINT_URL')
            
            self.settings = settings_module.settings
            
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
        """Test Bedrock embedding configuration."""
        try:
            # Check configuration
            bedrock_url = self.settings.bedrock_endpoint.url
            embedding_model = self.settings.aws.embedding_model
            
            if not bedrock_url:
                self.log_result(
                    "Configuration",
                    False,
                    "BEDROCK_ENDPOINT_URL not configured",
                    {"bedrock_url": bedrock_url}
                )
                return False
            
            self.log_result(
                "Configuration",
                True,
                f"Configuration loaded successfully",
                {
                    "bedrock_url": bedrock_url[:50] + "..." if len(bedrock_url) > 50 else bedrock_url,
                    "embedding_model": embedding_model,
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
    
    async def test_service_initialization(self):
        """Test Bedrock service initialization."""
        try:
            from text_to_sql_rag.services.enhanced_bedrock_service import EnhancedBedrockService
            
            self.bedrock_service = EnhancedBedrockService(
                endpoint_url=self.settings.bedrock_endpoint.url,
                embedding_model=self.settings.aws.embedding_model,
                llm_model=self.settings.aws.llm_model,
                verify_ssl=self.settings.bedrock_endpoint.verify_ssl,
                ssl_cert_file=self.settings.bedrock_endpoint.ssl_cert_file,
                ssl_key_file=self.settings.bedrock_endpoint.ssl_key_file,
                ssl_ca_file=self.settings.bedrock_endpoint.ssl_ca_file,
                http_auth_username=self.settings.bedrock_endpoint.http_auth_username,
                http_auth_password=self.settings.bedrock_endpoint.http_auth_password
            )
            
            service_info = self.bedrock_service.get_service_info()
            
            self.log_result(
                "Service Initialization",
                True,
                f"Bedrock service initialized successfully",
                service_info
            )
            return True
            
        except Exception as e:
            self.log_result(
                "Service Initialization",
                False,
                f"Service initialization failed: {e}"
            )
            return False
    
    def test_configuration_check(self):
        """Test Bedrock service configuration check (no API calls)."""
        try:
            if not hasattr(self, 'bedrock_service') or self.bedrock_service is None:
                # Service hasn't been initialized yet, check settings
                is_configured = bool(self.settings.bedrock_endpoint.url)
            else:
                is_configured = self.bedrock_service.is_configured()
            
            if is_configured:
                self.log_result(
                    "Configuration Check",
                    True,
                    "Bedrock service is properly configured"
                )
            else:
                self.log_result(
                    "Configuration Check",
                    False,
                    "Bedrock service configuration check failed"
                )
            
            return is_configured
            
        except Exception as e:
            self.log_result(
                "Configuration Check",
                False,
                f"Configuration check error: {e}"
            )
            return False
    
    def test_single_embedding(self):
        """Test generating a single embedding."""
        try:
            test_text = "This is a test document for embedding generation."
            
            embedding = self.bedrock_service.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                self.log_result(
                    "Single Embedding",
                    True,
                    f"Successfully generated embedding",
                    {
                        "text": test_text,
                        "embedding_dimension": len(embedding),
                        "embedding_type": type(embedding[0]).__name__,
                        "sample_values": embedding[:3]
                    }
                )
                return True
            else:
                self.log_result(
                    "Single Embedding",
                    False,
                    "Generated embedding is empty"
                )
                return False
            
        except Exception as e:
            self.log_result(
                "Single Embedding",
                False,
                f"Single embedding failed: {e}"
            )
            return False
    
    def test_batch_embeddings(self):
        """Test generating batch embeddings."""
        try:
            test_texts = [
                "First test document for batch embedding.",
                "Second document with different content.",
                "Third document to complete the batch test."
            ]
            
            embeddings = self.bedrock_service.get_embeddings_batch(test_texts, batch_size=2)
            
            if embeddings and len(embeddings) == len(test_texts):
                # Check consistency of dimensions
                dimensions = [len(emb) for emb in embeddings]
                consistent_dims = all(dim == dimensions[0] for dim in dimensions)
                
                self.log_result(
                    "Batch Embeddings",
                    True,
                    f"Successfully generated batch embeddings",
                    {
                        "text_count": len(test_texts),
                        "embedding_count": len(embeddings),
                        "dimensions": dimensions,
                        "consistent_dimensions": consistent_dims,
                        "sample_embedding_preview": embeddings[0][:3]
                    }
                )
                return True
            else:
                self.log_result(
                    "Batch Embeddings",
                    False,
                    "Batch embedding failed or returned wrong count",
                    {
                        "expected": len(test_texts),
                        "received": len(embeddings) if embeddings else 0
                    }
                )
                return False
            
        except Exception as e:
            self.log_result(
                "Batch Embeddings",
                False,
                f"Batch embeddings failed: {e}"
            )
            return False
    
    def test_embedding_consistency(self):
        """Test embedding consistency for the same text."""
        try:
            test_text = "Consistency test document for embedding stability."
            
            embedding1 = self.bedrock_service.get_embedding(test_text)
            embedding2 = self.bedrock_service.get_embedding(test_text)
            
            if embedding1 and embedding2:
                # Calculate similarity (cosine similarity)
                embedding1_np = np.array(embedding1)
                embedding2_np = np.array(embedding2)
                
                dot_product = np.dot(embedding1_np, embedding2_np)
                norms = np.linalg.norm(embedding1_np) * np.linalg.norm(embedding2_np)
                similarity = dot_product / norms if norms > 0 else 0
                
                is_consistent = similarity > 0.99  # Should be nearly identical
                
                self.log_result(
                    "Embedding Consistency",
                    is_consistent,
                    f"Embedding consistency test {'passed' if is_consistent else 'failed'}",
                    {
                        "similarity": f"{similarity:.6f}",
                        "embedding1_dim": len(embedding1),
                        "embedding2_dim": len(embedding2),
                        "expected_similarity": "> 0.99"
                    }
                )
                return is_consistent
            else:
                self.log_result(
                    "Embedding Consistency",
                    False,
                    "Failed to generate embeddings for consistency test"
                )
                return False
            
        except Exception as e:
            self.log_result(
                "Embedding Consistency",
                False,
                f"Embedding consistency test failed: {e}"
            )
            return False
    
    def test_dimension_detection(self):
        """Test embedding dimension detection."""
        try:
            # Generate an embedding first
            self.bedrock_service.get_embedding("Test for dimension detection")
            
            dimension = self.bedrock_service.get_embedding_dimension()
            
            if dimension and dimension > 0:
                self.log_result(
                    "Dimension Detection",
                    True,
                    f"Successfully detected embedding dimension",
                    {
                        "detected_dimension": dimension,
                        "expected_range": "512-4096 (typical for modern models)"
                    }
                )
                return True
            else:
                self.log_result(
                    "Dimension Detection",
                    False,
                    "Failed to detect embedding dimension"
                )
                return False
            
        except Exception as e:
            self.log_result(
                "Dimension Detection",
                False,
                f"Dimension detection failed: {e}"
            )
            return False
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        try:
            # Test with empty text
            try:
                embedding = self.bedrock_service.get_embedding("")
                empty_text_handled = len(embedding) == 0  # Should return empty or handle gracefully
            except Exception:
                empty_text_handled = True  # Exception is also acceptable
            
            # Test with very long text
            long_text = "This is a very long text. " * 1000  # ~27,000 characters
            try:
                embedding = self.bedrock_service.get_embedding(long_text)
                long_text_handled = len(embedding) > 0
            except Exception:
                long_text_handled = True  # Exception is acceptable for overly long text
            
            overall_success = empty_text_handled and long_text_handled
            
            self.log_result(
                "Error Handling",
                overall_success,
                f"Error handling test {'passed' if overall_success else 'failed'}",
                {
                    "empty_text_handled": empty_text_handled,
                    "long_text_handled": long_text_handled,
                    "long_text_length": len(long_text)
                }
            )
            return overall_success
            
        except Exception as e:
            self.log_result(
                "Error Handling",
                False,
                f"Error handling test failed: {e}"
            )
            return False
    
    async def run_all_tests(self):
        """Run all Bedrock embedding tests."""
        print("Starting Bedrock Embedding Tests")
        print("=" * 50)
        
        # Sync tests first
        tests_sync = [
            self.test_import_services,
            self.test_configuration,
            self.test_configuration_check
        ]
        
        # Async tests (only service initialization is still async)
        tests_async = [
            self.test_service_initialization
        ]
        
        # Sync tests (after service initialization)
        tests_sync_after = [
            self.test_single_embedding,
            self.test_batch_embeddings,
            self.test_embedding_consistency,
            self.test_dimension_detection,
            self.test_error_handling
        ]
        
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
        
        # Run sync tests after service initialization
        for test in tests_sync_after:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"FAIL: {test.__name__} - Unexpected error: {e}")
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("All Bedrock embedding tests passed!")
        else:
            print(f"{total - passed} test(s) failed")
        
        return passed == total


async def main():
    """Main function to run Bedrock embedding tests."""
    print("Bedrock Embedding Test Suite")
    print("This tests the actual Bedrock embedding functionality using application services")
    print()
    
    tester = BedrockEmbeddingTest()
    success = await tester.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)