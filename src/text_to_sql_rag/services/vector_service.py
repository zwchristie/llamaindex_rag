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
from llama_index.vector_stores.opensearch import OpenSearchVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from opensearchpy import OpenSearch
import ssl

from ..config.settings import settings
from ..utils.content_processor import ContentProcessor
from .bedrock_service import BedrockEmbeddingService, BedrockLLMService

logger = structlog.get_logger(__name__)


class LlamaIndexVectorService:
    """Service for managing documents using LlamaIndex with OpenSearch vector store."""
    
    def __init__(self):
        self.client = self._create_opensearch_client()
        self.index_name = settings.opensearch.index_name
        self.vector_field = settings.opensearch.vector_field
        self.vector_size = settings.opensearch.vector_size
        
        # Initialize content processor and bedrock services
        self.content_processor = ContentProcessor()
        self.bedrock_embedding = BedrockEmbeddingService()
        self.bedrock_llm = BedrockLLMService()
        
        # Initialize LlamaIndex components
        self._setup_llamaindex()
        
        # Initialize vector store and index
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
    
    def _create_opensearch_client(self) -> OpenSearch:
        """Create OpenSearch client with proper configuration."""
        try:
            # Configure connection based on settings
            client_config = {
                'hosts': [{'host': settings.opensearch.host, 'port': settings.opensearch.port}],
                'use_ssl': settings.opensearch.use_ssl,
                'verify_certs': settings.opensearch.verify_certs,
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            # Add authentication if provided
            if settings.opensearch.username and settings.opensearch.password:
                client_config['http_auth'] = (
                    settings.opensearch.username, 
                    settings.opensearch.password
                )
            
            # Configure SSL context if needed
            if settings.opensearch.use_ssl and not settings.opensearch.verify_certs:
                client_config['ssl_context'] = ssl.create_default_context()
                client_config['ssl_context'].check_hostname = False
                client_config['ssl_context'].verify_mode = ssl.CERT_NONE
            
            client = OpenSearch(**client_config)
            
            # Test connection
            info = client.info()
            logger.info(
                "Connected to OpenSearch", 
                host=settings.opensearch.host, 
                port=settings.opensearch.port,
                version=info.get('version', {}).get('number', 'unknown')
            )
            return client
            
        except Exception as e:
            logger.error("Failed to connect to OpenSearch", error=str(e))
            raise
    
    def _setup_llamaindex(self) -> None:
        """Setup LlamaIndex global settings using bedrock services."""
        try:
            # Configure embedding model - always use Bedrock for embeddings
            if settings.aws.use_profile and settings.aws.profile_name:
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
                if settings.aws.use_profile and settings.aws.profile_name:
                    # Use AWS profile for LLM
                    llm = Bedrock(
                        model=settings.aws.llm_model,
                        region_name=settings.aws.region,
                        profile_name=settings.aws.profile_name,
                        temperature=0.1,
                        max_tokens=2048
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
                        max_tokens=2048
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
    
    def _create_vector_store(self) -> OpenSearchVectorStore:
        """Create OpenSearch vector store for LlamaIndex."""
        try:
            vector_store = OpenSearchVectorStore(
                client=self.client,
                index_name=self.index_name,
                vector_field=self.vector_field,
                text_field="content",
                metadata_field="metadata",
                dim=self.vector_size
            )
            
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
            # Get index statistics from OpenSearch
            stats = self.client.indices.stats(index=self.index_name)
            index_stats = stats.get("indices", {}).get(self.index_name, {})
            
            return {
                "index_name": self.index_name,
                "documents_count": index_stats.get("total", {}).get("docs", {}).get("count", 0),
                "index_size": index_stats.get("total", {}).get("store", {}).get("size_in_bytes", 0),
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
            # Check OpenSearch connection
            cluster_health = self.client.cluster.health()
            
            # Check if index exists and is accessible
            if self.client.indices.exists(index=self.index_name):
                return cluster_health.get("status") in ["green", "yellow"]
            
            return True  # Cluster is healthy even if index doesn't exist yet
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False