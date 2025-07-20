"""LlamaIndex-based vector store service with Qdrant integration."""

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
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..config.settings import settings

logger = structlog.get_logger(__name__)


class LlamaIndexVectorService:
    """Service for managing documents using LlamaIndex with Qdrant vector store."""
    
    def __init__(self):
        self.client = self._create_qdrant_client()
        self.collection_name = settings.qdrant.collection_name
        self.vector_size = settings.qdrant.vector_size
        
        # Initialize LlamaIndex components
        self._setup_llamaindex()
        
        # Initialize vector store and index
        self.vector_store = self._create_vector_store()
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create or load index
        self.index = self._get_or_create_index()
        
        # Setup retrievers for hybrid search
        self.retrievers = self._setup_retrievers()
    
    def _create_qdrant_client(self) -> QdrantClient:
        """Create Qdrant client with proper configuration."""
        try:
            if settings.qdrant.api_key:
                client = QdrantClient(
                    url=f"https://{settings.qdrant.host}:{settings.qdrant.port}",
                    api_key=settings.qdrant.api_key
                )
            else:
                client = QdrantClient(
                    host=settings.qdrant.host,
                    port=settings.qdrant.port
                )
            
            logger.info("Connected to Qdrant", host=settings.qdrant.host, port=settings.qdrant.port)
            return client
            
        except Exception as e:
            logger.error("Failed to connect to Qdrant", error=str(e))
            raise
    
    def _setup_llamaindex(self) -> None:
        """Setup LlamaIndex global settings."""
        try:
            # Configure embedding model
            embed_model = BedrockEmbedding(
                model_name=settings.aws.embedding_model,
                region_name=settings.aws.region,
                aws_access_key_id=settings.aws.access_key_id,
                aws_secret_access_key=settings.aws.secret_access_key,
                aws_session_token=settings.aws.session_token
            )
            
            # Configure LLM
            llm = Bedrock(
                model=settings.aws.llm_model,
                region_name=settings.aws.region,
                aws_access_key_id=settings.aws.access_key_id,
                aws_secret_access_key=settings.aws.secret_access_key,
                aws_session_token=settings.aws.session_token,
                temperature=0.1,
                max_tokens=2048
            )
            
            # Set global settings
            Settings.embed_model = embed_model
            Settings.llm = llm
            Settings.chunk_size = settings.app.chunk_size
            Settings.chunk_overlap = settings.app.chunk_overlap
            
            logger.info("LlamaIndex settings configured successfully")
            
        except Exception as e:
            logger.error("Failed to setup LlamaIndex", error=str(e))
            raise
    
    def _create_vector_store(self) -> QdrantVectorStore:
        """Create Qdrant vector store for LlamaIndex."""
        try:
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                enable_hybrid=True,  # Enable hybrid search
                batch_size=20
            )
            
            logger.info("Created Qdrant vector store", collection_name=self.collection_name)
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
            
            # Hybrid retriever (if supported by vector store)
            if hasattr(self.vector_store, 'enable_hybrid') and self.vector_store.enable_hybrid:
                from llama_index.core.retrievers import QueryFusionRetriever
                retrievers["hybrid"] = QueryFusionRetriever(
                    retrievers=[retrievers["vector"]],
                    similarity_top_k=settings.app.similarity_top_k,
                    num_queries=3  # Generate multiple query variants
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
        """Add document to the index."""
        try:
            # Create LlamaIndex document with metadata
            doc_metadata = {
                "document_id": str(document_id),
                "document_type": document_type,
                **metadata
            }
            
            document = LlamaDocument(
                text=content,
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
            
            # Apply filters if needed
            if document_type or document_ids:
                filters = self._build_metadata_filters(document_type, document_ids)
                if hasattr(retriever, 'filters'):
                    retriever.filters = filters
            
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
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            # Get collection info from Qdrant
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "retrievers_available": list(self.retrievers.keys()),
                "settings": {
                    "chunk_size": settings.app.chunk_size,
                    "chunk_overlap": settings.app.chunk_overlap,
                    "similarity_top_k": settings.app.similarity_top_k
                }
            }
            
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            # Check Qdrant connection
            collections = self.client.get_collections()
            
            # Check if index is accessible
            if self.index:
                return True
            
            return False
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False