"""LlamaIndex-based vector store service with OpenSearch integration."""

from typing import List, Dict, Any, Optional, Tuple
import structlog
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex, 
    Document as LlamaDocument,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient
)
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM

import ssl

from ..config.settings import settings
from ..utils.content_processor import ContentProcessor
from .bedrock_service import BedrockEmbeddingService, BedrockLLMService

logger = structlog.get_logger(__name__)


class CustomBedrockEmbedding(BaseEmbedding):
    """Custom embedding wrapper for inference profile ARNs."""
    
    def __init__(self, bedrock_service: 'BedrockEmbeddingService', **kwargs):
        # Store the service before calling super() to avoid Pydantic validation issues
        if bedrock_service is None:
            raise ValueError("bedrock_service cannot be None")
        super().__init__(**kwargs)
        # Set this after super() to ensure it's not lost during Pydantic validation
        object.__setattr__(self, '_bedrock_service', bedrock_service)
    
    def __getstate__(self):
        """Custom pickling to preserve bedrock service."""
        state = self.__dict__.copy()
        # Ensure bedrock_service is preserved
        if hasattr(self, '_bedrock_service'):
            state['_bedrock_service'] = self._bedrock_service
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore bedrock service."""
        self.__dict__.update(state)
        # Recreate bedrock service if not present
        if '_bedrock_service' not in state or state['_bedrock_service'] is None:
            from .bedrock_service import BedrockEmbeddingService
            object.__setattr__(self, '_bedrock_service', BedrockEmbeddingService())
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query."""
        if not hasattr(self, '_bedrock_service') or self._bedrock_service is None:
            from .bedrock_service import BedrockEmbeddingService
            object.__setattr__(self, '_bedrock_service', BedrockEmbeddingService())
        return self._bedrock_service.get_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if not hasattr(self, '_bedrock_service') or self._bedrock_service is None:
            from .bedrock_service import BedrockEmbeddingService
            object.__setattr__(self, '_bedrock_service', BedrockEmbeddingService())
        return self._bedrock_service.get_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for query."""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding for text."""
        return self._get_text_embedding(text)


class CustomBedrockLLM(LLM):
    """Custom LLM wrapper for inference profile ARNs."""
    
    def __init__(self, bedrock_service: 'BedrockLLMService', **kwargs):
        # Store the service before calling super() to avoid Pydantic validation issues
        self._bedrock_service = bedrock_service
        super().__init__(**kwargs)
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt."""
        return self._bedrock_service.generate_text(prompt, **kwargs)
    
    def stream_complete(self, prompt: str, **kwargs):
        """Stream complete - not implemented for simplicity."""
        # For now, just return the complete response
        response = self.complete(prompt, **kwargs)
        yield response
    
    def chat(self, messages, **kwargs) -> str:
        """Chat interface - convert messages to prompt."""
        # Simple conversion from messages to prompt
        prompt = ""
        for message in messages:
            if hasattr(message, 'content'):
                prompt += f"{message.content}\n"
            else:
                prompt += f"{message}\n"
        return self.complete(prompt, **kwargs)
    
    def stream_chat(self, messages, **kwargs):
        """Stream chat interface."""
        response = self.chat(messages, **kwargs)
        yield response
    
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Async complete."""
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs):
        """Async stream complete."""
        response = await self.acomplete(prompt, **kwargs)
        yield response
    
    async def achat(self, messages, **kwargs) -> str:
        """Async chat."""
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages, **kwargs):
        """Async stream chat."""
        response = await self.achat(messages, **kwargs)
        yield response
    
    @property
    def metadata(self):
        """Return metadata about the LLM."""
        return {"model_name": "custom_bedrock"}


def _is_inference_profile_arn(model_id: str) -> bool:
    """Check if model ID is an inference profile ARN."""
    return model_id.startswith("arn:aws:bedrock:") and "application-inference-profile" in model_id


class LlamaIndexVectorService:
    """Service for managing documents using LlamaIndex with OpenSearch vector store."""
    
    def __init__(self):
        self.index_name = settings.opensearch.index_name
        self.vector_field = settings.opensearch.vector_field
        self.vector_size = settings.opensearch.vector_size
        
        # Initialize content processor and bedrock services
        self.content_processor = ContentProcessor()
        self.bedrock_embedding = BedrockEmbeddingService()
        self.bedrock_llm = BedrockLLMService()
        
        # Initialize LlamaIndex components
        self._setup_llamaindex()
        
        # Initialize OpensearchVectorClient and vector store
        self.opensearch_client = self._create_opensearch_vector_client()
        self.vector_store = self._create_vector_store()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create or load index
        self.index = self._get_or_create_index()
        
        # Setup retrievers for hybrid search
        self.retrievers = self._setup_retrievers()
        
        # Create query engine property for LangGraph agent (lazy initialization)
        self._query_engine = None
    
    @property 
    def query_engine(self):
        """Lazy initialization of query engine for LangGraph agent."""
        if self._query_engine is None:
            retriever = self.retrievers.get("hybrid", self.retrievers["vector"])
            self._query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_mode="compact"
            )
        return self._query_engine
    
    def _create_opensearch_vector_client(self) -> OpensearchVectorClient:
        """Create OpensearchVectorClient with proper configuration."""
        try:
            # Build endpoint URL
            protocol = "https" if settings.opensearch.use_ssl else "http"
            endpoint = f"{protocol}://{settings.opensearch.host}:{settings.opensearch.port}"
            
            # Example hostnames for reference:
            # Local development: "http://localhost:9200"
            # AWS OpenSearch Service: "https://search-my-domain-abc123.us-east-1.es.amazonaws.com"
            # Self-hosted: "https://opensearch.example.com:9200"
            
            # Create OpensearchVectorClient - separate OpenSearch client kwargs from LlamaIndex kwargs
            llamaindex_kwargs = {
                'endpoint': endpoint,
                'index': self.index_name,
                'dim': self.vector_size,
                'embedding_field': self.vector_field,
                'text_field': 'content',
                'metadata_field': 'metadata'
            }
            
            # OpenSearch client kwargs that get passed through
            opensearch_client_kwargs = {}
            
            # Add authentication if provided
            if settings.opensearch.username and settings.opensearch.password:
                opensearch_client_kwargs['http_auth'] = (
                    settings.opensearch.username,
                    settings.opensearch.password
                )
            
            # Configure SSL settings
            if settings.opensearch.use_ssl:
                opensearch_client_kwargs['use_ssl'] = True
                if not settings.opensearch.verify_certs:
                    opensearch_client_kwargs['verify_certs'] = False
                    opensearch_client_kwargs['ssl_show_warn'] = False
                    opensearch_client_kwargs['ssl_assert_hostname'] = False
            
            # Combine all kwargs
            client_kwargs = {**llamaindex_kwargs, **opensearch_client_kwargs}
            
            # Debug logging to see what we're passing
            logger.info(
                "Creating OpensearchVectorClient with parameters",
                endpoint=endpoint,
                index=self.index_name,
                vector_size=self.vector_size,
                has_auth=bool(settings.opensearch.username and settings.opensearch.password),
                use_ssl=settings.opensearch.use_ssl,
                verify_certs=settings.opensearch.verify_certs,
                client_kwargs_keys=list(client_kwargs.keys())
            )
            
            client = OpensearchVectorClient(**client_kwargs)
            
            logger.info(
                "Created OpensearchVectorClient successfully",
                endpoint=endpoint,
                index=self.index_name,
                vector_size=self.vector_size
            )
            return client
            
        except Exception as e:
            logger.error("Failed to create OpensearchVectorClient", error=str(e))
            raise
    
    def _setup_llamaindex(self) -> None:
        """Setup LlamaIndex global settings using bedrock services."""
        try:
            # Configure embedding model - always use Bedrock for embeddings
            # Check if we're using inference profile ARNs
            if _is_inference_profile_arn(settings.aws.embedding_model):
                # Use custom embedding wrapper for inference profiles
                embed_model = CustomBedrockEmbedding(self.bedrock_embedding)
                logger.info("Using custom embedding wrapper for inference profile ARN")
            elif settings.aws.use_profile and settings.aws.profile_name:
                # Use AWS profile for embeddings
                embed_model = BedrockEmbedding(
                    model_name=settings.aws.embedding_model,
                    region_name=settings.aws.region,
                    profile_name=settings.aws.profile_name
                )
            else:
                # Use explicit credentials or default chain
                embed_model = BedrockEmbedding(
                    model_name=settings.aws.embedding_model,
                    region_name=settings.aws.region,
                    aws_access_key_id=settings.aws.access_key_id,
                    aws_secret_access_key=settings.aws.secret_access_key,
                    aws_session_token=settings.aws.session_token
                )
            
            # Configure LLM - only use Bedrock for LlamaIndex if using Bedrock provider
            llm = None
            if settings.is_using_bedrock():
                # Check if we're using inference profile ARNs for LLM
                if _is_inference_profile_arn(settings.aws.llm_model):
                    # Use custom LLM wrapper for inference profiles
                    llm = CustomBedrockLLM(self.bedrock_llm)
                    logger.info("Using custom LLM wrapper for inference profile ARN")
                elif settings.aws.use_profile and settings.aws.profile_name:
                    # Use AWS profile for LLM
                    llm = Bedrock(
                        model=settings.aws.llm_model,
                        region_name=settings.aws.region,
                        profile_name=settings.aws.profile_name,
                        temperature=0.1,
                        max_tokens=2048,
                        context_size=200000
                    )
                else:
                    # Use explicit credentials or default chain
                    llm = Bedrock(
                        model=settings.aws.llm_model,
                        region_name=settings.aws.region,
                        aws_access_key_id=settings.aws.access_key_id,
                        aws_secret_access_key=settings.aws.secret_access_key,
                        aws_session_token=settings.aws.session_token,
                        temperature=0.1,
                        max_tokens=2048,
                        context_size=200000
                    )
            else:
                # For custom LLM provider, use a simple OpenAI-like wrapper or None
                # LlamaIndex will fall back to default behavior
                logger.info("Using custom LLM provider, LlamaIndex LLM set to None")
                llm = None
            
            # Set global settings
            Settings.embed_model = embed_model
            if llm is not None:
                Settings.llm = llm
            Settings.chunk_size = settings.app.chunk_size
            Settings.chunk_overlap = settings.app.chunk_overlap
            
            logger.info("LlamaIndex settings configured successfully")
            
        except Exception as e:
            logger.error("Failed to setup LlamaIndex", error=str(e))
            raise
    
    def _create_vector_store(self) -> OpensearchVectorStore:
        """Create OpenSearch vector store for LlamaIndex using OpensearchVectorClient."""
        try:
            vector_store = OpensearchVectorStore(self.opensearch_client)
            
            logger.info("Created OpenSearch vector store", index_name=self.index_name)
            return vector_store
            
        except Exception as e:
            logger.error("Failed to create vector store", error=str(e))
            raise
    
    def _get_or_create_index(self) -> VectorStoreIndex:
        """Get existing index or create new one."""
        try:
            # Try to create index from existing vector store
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    storage_context=self.storage_context
                )
                logger.info("Loaded existing vector index")
            except:
                # Create new empty index
                index = VectorStoreIndex(
                    nodes=[],
                    storage_context=self.storage_context
                )
                logger.info("Created new vector index")
            
            return index
            
        except Exception as e:
            logger.error("Failed to get or create index", error=str(e))
            raise
    
    def _setup_retrievers(self) -> Dict[str, Any]:
        """Setup different retriever types for hybrid search."""
        retrievers = {}
        
        try:
            # Vector retriever
            retrievers["vector"] = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=settings.app.similarity_top_k
            )
            
            # For OpenSearch, we can use the same vector retriever
            # OpenSearch supports hybrid search natively through its query DSL
            retrievers["hybrid"] = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=settings.app.similarity_top_k
            )
            
            logger.info("Setup retrievers", types=list(retrievers.keys()))
            return retrievers
            
        except Exception as e:
            logger.error("Failed to setup retrievers", error=str(e))
            return {"vector": retrievers.get("vector")}
    
    def add_document(
        self,
        document_id: int,
        content: str,
        metadata: Dict[str, Any],
        document_type: str
    ) -> bool:
        """Add document to the index with JSON to Dolphin format conversion."""
        try:
            # Convert JSON documents to Dolphin format for better vectorization
            processed_content = content
            if self.content_processor.is_json_content(content):
                from ..models.simple_models import DocumentType as DocType
                doc_type_enum = DocType.SCHEMA if document_type.lower() == "schema" else DocType.REPORT
                processed_content = self.content_processor.convert_json_to_dolphin_format(
                    content, doc_type_enum
                )
                logger.info(
                    "Converted JSON document to Dolphin format",
                    document_id=document_id,
                    original_length=len(content),
                    processed_length=len(processed_content)
                )
            
            # Create LlamaIndex document with metadata
            doc_metadata = {
                "document_id": str(document_id),
                "document_type": document_type,
                "is_json_converted": self.content_processor.is_json_content(content),
                **metadata
            }
            
            document = LlamaDocument(
                text=processed_content,
                metadata=doc_metadata,
                id_=f"doc_{document_id}"
            )
            
            # Use sentence splitter for better chunking
            splitter = SentenceSplitter(
                chunk_size=settings.app.chunk_size,
                chunk_overlap=settings.app.chunk_overlap
            )
            
            # Parse document into nodes
            nodes = splitter.get_nodes_from_documents([document])
            
            # Add metadata to each node
            for i, node in enumerate(nodes):
                node.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(nodes)
                })
            
            # Insert nodes into index
            self.index.insert_nodes(nodes)
            
            logger.info(
                "Added document to index",
                document_id=document_id,
                num_chunks=len(nodes),
                document_type=document_type
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to add document to index", document_id=document_id, error=str(e))
            return False
    
    def search_similar(
        self,
        query: str,
        retriever_type: str = "hybrid",
        similarity_top_k: Optional[int] = None,
        document_type: Optional[str] = None,
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content using specified retriever."""
        try:
            # Select retriever
            retriever = self.retrievers.get(retriever_type, self.retrievers["vector"])
            
            # Update similarity_top_k if provided
            if similarity_top_k:
                retriever.similarity_top_k = similarity_top_k
            
            # Apply filters if needed (OpenSearch supports native filtering)
            if document_type or document_ids:
                filters = self._build_metadata_filters(document_type, document_ids)
                # Note: Filtering implementation may need adjustment based on LlamaIndex OpenSearch integration
            
            # Perform retrieval
            nodes = retriever.retrieve(query)
            
            # Format results
            results = []
            for node in nodes:
                results.append({
                    "id": node.node_id,
                    "score": getattr(node, 'score', 1.0),
                    "content": node.text,
                    "metadata": node.metadata,
                    "document_id": node.metadata.get("document_id"),
                    "document_type": node.metadata.get("document_type")
                })
            
            logger.info(
                "Performed similarity search",
                query_length=len(query),
                num_results=len(results),
                retriever_type=retriever_type
            )
            
            return results
            
        except Exception as e:
            logger.error("Failed to search", error=str(e))
            return []
    
    def query_with_context(
        self,
        query: str,
        retriever_type: str = "hybrid",
        response_mode: str = "compact"
    ) -> Dict[str, Any]:
        """Query with generated response using retrieved context."""
        try:
            # Select retriever
            retriever = self.retrievers.get(retriever_type, self.retrievers["vector"])
            
            # Create query engine
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                response_mode=response_mode
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        "document_id": node.metadata.get("document_id"),
                        "document_type": node.metadata.get("document_type"),
                        "content_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": getattr(node, 'score', 1.0)
                    })
            
            result = {
                "query": query,
                "response": str(response),
                "sources": sources,
                "retriever_type": retriever_type
            }
            
            logger.info("Generated response with context", query_length=len(query), num_sources=len(sources))
            return result
            
        except Exception as e:
            logger.error("Failed to query with context", error=str(e))
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "sources": [],
                "retriever_type": retriever_type
            }
    
    def delete_document(self, document_id: int) -> bool:
        """Delete document from the index."""
        try:
            # Get all nodes for this document
            nodes = self.search_similar(
                query="",  # Empty query to get all
                document_ids=[document_id]
            )
            
            # Delete nodes by ID
            node_ids = [node["id"] for node in nodes]
            if node_ids:
                self.index.delete_nodes(node_ids)
                
                logger.info("Deleted document from index", document_id=document_id, num_nodes=len(node_ids))
                return True
            
            return True  # No nodes to delete
            
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            return False
    
    def update_document(
        self,
        document_id: int,
        content: str,
        metadata: Dict[str, Any],
        document_type: str
    ) -> bool:
        """Update document in the index."""
        try:
            # Delete existing document
            self.delete_document(document_id)
            
            # Add updated document
            return self.add_document(document_id, content, metadata, document_type)
            
        except Exception as e:
            logger.error("Failed to update document", document_id=document_id, error=str(e))
            return False
    
    def _build_metadata_filters(
        self,
        document_type: Optional[str] = None,
        document_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Build metadata filters for retrieval."""
        filters = {}
        
        if document_type:
            filters["document_type"] = document_type
        
        if document_ids:
            filters["document_id"] = [str(doc_id) for doc_id in document_ids]
        
        return filters
    
    def get_document_info(self, document_id: int) -> Dict[str, Any]:
        """Get information about a document's vectors."""
        try:
            # Search for documents with this ID
            results = self.search_similar(
                query="*",  # Match all
                document_ids=[document_id],
                similarity_top_k=100
            )
            
            info = {
                "document_id": document_id,
                "num_chunks": len(results),
                "metadata": results[0]["metadata"] if results else {}
            }
            
            return info
            
        except Exception as e:
            logger.error("Failed to get document info", document_id=document_id, error=str(e))
            return {"document_id": document_id, "num_chunks": 0, "metadata": {}}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            # Return basic stats without querying OpenSearch directly
            # since OpensearchVectorClient doesn't expose the underlying client
            return {
                "index_name": self.index_name,
                "documents_count": "N/A (requires direct OpenSearch access)",
                "index_size": "N/A (requires direct OpenSearch access)",
                "retrievers_available": list(self.retrievers.keys()),
                "settings": {
                    "chunk_size": settings.app.chunk_size,
                    "chunk_overlap": settings.app.chunk_overlap,
                    "similarity_top_k": settings.app.similarity_top_k,
                    "vector_size": self.vector_size
                }
            }
            
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            # Simple health check - try to perform a basic operation
            # Since OpensearchVectorClient doesn't expose the underlying client,
            # we'll just check if our components are initialized
            if (hasattr(self, 'opensearch_client') and self.opensearch_client is not None and
                hasattr(self, 'index') and self.index is not None and
                hasattr(self, 'retrievers') and self.retrievers):
                return True
            return False
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False