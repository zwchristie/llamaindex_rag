#!/usr/bin/env python3
"""Test complete RAG pipeline functionality."""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
import motor.motor_asyncio
from opensearchpy import AsyncOpenSearch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RAGPipelineTests:
    """Test suite for complete RAG pipeline."""
    
    def __init__(self):
        # Database connections
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
        self.database_name = os.getenv("MONGODB_DATABASE", "text_to_sql_rag")
        
        # OpenSearch connection
        self.opensearch_host = os.getenv("OPENSEARCH_HOST", "localhost")
        self.opensearch_port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        
        # Bedrock configuration
        self.bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT_URL", "https://8v1n9dbomk.execute-api.us-east-1.amazonaws.com/testaccess")
        self.embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        self.llm_model = os.getenv("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        self.use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"
        
        # Service instances
        self.mongo_client = None
        self.db = None
        self.opensearch_client = None
        self.embedding_service = None
        self.vector_service = None
        self.bedrock_service = None
        
        self.test_results = []
    
    async def setup(self):
        """Initialize all services for testing."""
        try:
            # MongoDB connection
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.db = self.mongo_client[self.database_name]
            await self.mongo_client.server_info()
            logger.info("✅ MongoDB connected")
            
            # OpenSearch connection
            self.opensearch_client = AsyncOpenSearch(
                hosts=[{"host": self.opensearch_host, "port": self.opensearch_port}],
                http_auth=None, use_ssl=False, verify_certs=False
            )
            cluster_info = await self.opensearch_client.info()
            logger.info(f"✅ OpenSearch connected: {cluster_info['version']['number']}")
            
            # Import services
            from text_to_sql_rag.services.view_service import ViewService
            from text_to_sql_rag.services.embedding_service import EmbeddingService, VectorService
            from text_to_sql_rag.services.bedrock_endpoint_service import BedrockEndpointService
            
            # Initialize services
            self.view_service = ViewService(self.db)
            self.embedding_service = EmbeddingService(
                endpoint_url=self.bedrock_endpoint,
                embedding_model=self.embedding_model,
                use_mock=self.use_mock
            )
            self.vector_service = VectorService(self.opensearch_client, "view_metadata", "embedding")
            self.bedrock_service = BedrockEndpointService(
                self.bedrock_endpoint, 
                self.embedding_model, 
                self.llm_model
            )
            
            logger.info("✅ All services initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Service setup failed: {e}")
            return False
    
    async def test_embedding_generation(self):
        """Test embedding generation from text."""
        test_name = "Embedding Generation"
        try:
            test_queries = [
                "user engagement metrics",
                "syndicate participation data", 
                "transaction summary information",
                "financial reporting views"
            ]
            
            successful_embeddings = 0
            
            for query in test_queries:
                try:
                    embedding = await self.embedding_service.get_embedding(query)
                    
                    if embedding and len(embedding) > 0:
                        successful_embeddings += 1
                        expected_dim = 1536 if self.use_mock else 1024
                        if len(embedding) == expected_dim:
                            logger.info(f"✅ {query}: {len(embedding)}-dim embedding generated")
                        else:
                            logger.warning(f"⚠️  {query}: Unexpected dimension {len(embedding)}, expected {expected_dim}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate embedding for '{query}': {e}")
            
            if successful_embeddings == len(test_queries):
                self.test_results.append({
                    "test": test_name,
                    "status": "PASS",
                    "message": f"All {len(test_queries)} embeddings generated successfully"
                })
            elif successful_embeddings > 0:
                self.test_results.append({
                    "test": test_name,
                    "status": "PARTIAL", 
                    "message": f"{successful_embeddings}/{len(test_queries)} embeddings generated"
                })
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "message": "No embeddings could be generated"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "message": f"Embedding test failed: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def test_vector_search(self):
        """Test vector similarity search."""
        test_name = "Vector Search"
        try:
            # Check if index exists and has documents
            try:
                index_info = await self.opensearch_client.indices.get("view_metadata")
                doc_count = await self.opensearch_client.count(index="view_metadata")
                
                if doc_count["count"] == 0:
                    self.test_results.append({
                        "test": test_name,
                        "status": "SKIP",
                        "message": "No documents in vector index - run data indexing first"
                    })
                    return
                    
                logger.info(f"📊 Index has {doc_count['count']} documents")
                
            except Exception as e:
                self.test_results.append({
                    "test": test_name,
                    "status": "SKIP", 
                    "message": f"Vector index not available: {str(e)}"
                })
                return
            
            # Test search queries
            search_queries = [
                ("user metrics", ["V_USER_METRICS"]),
                ("syndicate participation", ["V_TRANCHE_SYNDICATES"]), 
                ("allocation data", ["V_SYNDICATE_ALLOCATIONS"])
            ]
            
            successful_searches = 0
            
            for query, expected_views in search_queries:
                try:
                    # Generate query embedding
                    query_embedding = await self.embedding_service.get_embedding(query)
                    
                    # Search similar views
                    similar_views = await self.vector_service.search_similar_views(query_embedding, k=3)
                    
                    if similar_views:
                        successful_searches += 1
                        
                        # Check if expected views are found
                        found_views = [view.view_name for view, score in similar_views]
                        relevance_scores = [score for view, score in similar_views]
                        
                        logger.info(f"✅ '{query}' found {len(similar_views)} similar views:")
                        for view, score in similar_views:
                            logger.info(f"   - {view.view_name} (similarity: {score:.3f})")
                        
                        # Check if any expected views were found
                        expected_found = any(expected in found_views for expected in expected_views)
                        if expected_found:
                            logger.info(f"   ✅ Expected views found in results")
                        else:
                            logger.warning(f"   ⚠️  Expected views {expected_views} not found")
                    
                except Exception as e:
                    logger.error(f"❌ Search failed for '{query}': {e}")
            
            if successful_searches == len(search_queries):
                self.test_results.append({
                    "test": test_name,
                    "status": "PASS",
                    "message": f"All {len(search_queries)} searches completed successfully"
                })
            elif successful_searches > 0:
                self.test_results.append({
                    "test": test_name,
                    "status": "PARTIAL",
                    "message": f"{successful_searches}/{len(search_queries)} searches completed"
                })
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "message": "No searches completed successfully"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "message": f"Vector search test failed: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def test_context_retrieval(self):
        """Test context retrieval and formatting."""
        test_name = "Context Retrieval"
        try:
            # Test context building from retrieved views
            query = "user engagement and activity metrics"
            
            # Get embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Search for similar views
            similar_views = await self.vector_service.search_similar_views(query_embedding, k=2)
            
            if not similar_views:
                self.test_results.append({
                    "test": test_name,
                    "status": "SKIP",
                    "message": "No views found for context retrieval test"
                })
                return
            
            # Build context from retrieved views
            context_parts = []
            for view, score in similar_views:
                context_text = view.generate_full_text()
                context_parts.append(context_text)
                logger.info(f"📄 Retrieved context from {view.view_name} ({len(context_text)} chars)")
            
            # Combine contexts
            full_context = "\n\n".join(context_parts)
            
            # Validate context quality
            validations = [
                ("Context not empty", len(full_context) > 0),
                ("Multiple views included", len(similar_views) > 1),
                ("Context contains view names", any(view.view_name in full_context for view, _ in similar_views)),
                ("Context has structure info", any(word in full_context.lower() for word in ['column', 'table', 'view'])),
                ("Context length appropriate", 100 < len(full_context) < 10000)
            ]
            
            passed_validations = sum(1 for _, passed in validations if passed)
            
            for validation_name, passed in validations:
                if passed:
                    logger.info(f"✅ {validation_name}")
                else:
                    logger.warning(f"⚠️  {validation_name}")
            
            logger.info(f"📊 Context summary: {len(full_context)} characters, {len(similar_views)} views")
            
            if passed_validations == len(validations):
                self.test_results.append({
                    "test": test_name,
                    "status": "PASS",
                    "message": f"Context retrieval successful - {len(full_context)} chars from {len(similar_views)} views"
                })
            elif passed_validations > len(validations) // 2:
                self.test_results.append({
                    "test": test_name,
                    "status": "PARTIAL",
                    "message": f"{passed_validations}/{len(validations)} validations passed"
                })
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "message": f"Context quality issues: {passed_validations}/{len(validations)} validations passed"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "message": f"Context retrieval failed: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def test_llm_integration(self):
        """Test LLM text generation."""
        test_name = "LLM Integration"
        try:
            # Test simple text generation
            test_prompt = "Generate a brief explanation of what a database view is."
            
            response = await self.bedrock_service.generate_text(test_prompt)
            
            if response and len(response.strip()) > 0:
                logger.info(f"✅ LLM generated response ({len(response)} characters)")
                logger.info(f"📝 Sample response: {response[:100]}...")
                
                # Quality checks
                quality_checks = [
                    ("Response not empty", len(response.strip()) > 0),
                    ("Reasonable length", 20 < len(response) < 1000),
                    ("Contains relevant keywords", any(word in response.lower() for word in ['database', 'view', 'table', 'data'])),
                    ("Coherent text", not response.count('\\n') > len(response) / 10)  # Not too fragmented
                ]
                
                passed_checks = sum(1 for _, passed in quality_checks if passed)
                
                for check_name, passed in quality_checks:
                    if passed:
                        logger.info(f"✅ {check_name}")
                    else:
                        logger.warning(f"⚠️  {check_name}")
                
                if passed_checks == len(quality_checks):
                    self.test_results.append({
                        "test": test_name,
                        "status": "PASS",
                        "message": "LLM integration working correctly"
                    })
                else:
                    self.test_results.append({
                        "test": test_name,
                        "status": "PARTIAL",
                        "message": f"LLM responding but {passed_checks}/{len(quality_checks)} quality checks passed"
                    })
            else:
                self.test_results.append({
                    "test": test_name,
                    "status": "FAIL",
                    "message": "LLM returned empty or invalid response"
                })
                
        except Exception as e:
            self.test_results.append({
                "test": test_name,
                "status": "FAIL",
                "message": f"LLM integration failed: {str(e)}"
            })
            logger.error(f"❌ {test_name} failed: {e}")
    
    async def cleanup(self):
        """Clean up connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if self.opensearch_client:
            await self.opensearch_client.close()
        logger.info("🔚 Cleaned up connections")
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("RAG PIPELINE TEST RESULTS")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        partial = sum(1 for result in self.test_results if result["status"] == "PARTIAL")
        skipped = sum(1 for result in self.test_results if result["status"] == "SKIP")
        
        for result in self.test_results:
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌",
                "PARTIAL": "⚠️",
                "SKIP": "⏭️"
            }
            print(f"{status_icon[result['status']]} {result['test']}: {result['message']}")
        
        print(f"\nSummary: {passed} passed, {failed} failed, {partial} partial, {skipped} skipped")
        
        if failed == 0 and passed > 0:
            print("🎉 RAG pipeline is working correctly!")
            return True
        else:
            print(f"⚠️  Pipeline has issues - check logs for details")
            return False

async def main():
    """Run all RAG pipeline tests."""
    print("STARTING RAG PIPELINE TESTS")
    print("="*60)
    
    tester = RAGPipelineTests()
    
    try:
        # Setup
        if not await tester.setup():
            print("❌ Failed to setup test environment")
            return False
        
        # Run tests
        await tester.test_embedding_generation()
        await tester.test_vector_search()
        await tester.test_context_retrieval()
        await tester.test_llm_integration()
        
        # Print results
        success = tester.print_summary()
        return success
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)