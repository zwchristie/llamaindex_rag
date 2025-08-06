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
from .bedrock_endpoint_service import BedrockEndpointEmbeddingService, BedrockEndpointLLMWrapper

logger = structlog.get_logger(__name__)


class CustomBedrockEmbedding(BaseEmbedding):
    """Custom embedding wrapper for inference profile ARNs and endpoint services."""
    
    def __init__(self, endpoint_service=None, **kwargs):
        # Store the endpoint service before calling super() to avoid Pydantic validation issues
        if endpoint_service is None:
            raise ValueError("endpoint_service must be provided")
        super().__init__(**kwargs)
        # Set this after super() to ensure it's not lost during Pydantic validation
        object.__setattr__(self, '_endpoint_service', endpoint_service)
    
    def __getstate__(self):
        """Custom pickling to preserve services."""
        state = self.__dict__.copy()
        # Ensure endpoint service is preserved
        if hasattr(self, '_endpoint_service'):
            state['_endpoint_service'] = self._endpoint_service
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore services."""
        self.__dict__.update(state)
        # Endpoint service should be preserved in state - if not, something went wrong
        if '_endpoint_service' not in state or state['_endpoint_service'] is None:
            raise RuntimeError("Endpoint service not preserved during pickling - cannot restore embedding service")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query."""
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query provided for embedding")
            return [0.0] * 1024
        
        try:
            # Use endpoint service for embeddings (only supported approach)
            embedding = self._endpoint_service.get_embedding(query)
            
            if not embedding or len(embedding) == 0:
                logger.warning("Empty embedding returned from service")
                return [0.0] * 1024
            return embedding
        except Exception as e:
            logger.error("Failed to get query embedding", error=str(e), query_length=len(query))
            return [0.0] * 1024
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding")
            # Return a zero vector instead of failing
            return [0.0] * 1024  # Match the vector size
        
        try:
            # Use endpoint service for embeddings (only supported approach)
            embedding = self._endpoint_service.get_embedding(text)
            
            if not embedding or len(embedding) == 0:
                logger.warning("Empty embedding returned from service")
                return [0.0] * 1024
            return embedding
        except Exception as e:
            logger.error("Failed to get embedding", error=str(e), text_length=len(text))
            return [0.0] * 1024
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for query."""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding for text."""
        return self._get_text_embedding(text)


class CustomBedrockLLM(LLM):
    """Custom LLM wrapper for inference profile ARNs and endpoint services."""
    
    def __init__(self, endpoint_service=None, **kwargs):
        # Store the endpoint service before calling super() to avoid Pydantic validation issues
        if endpoint_service is None:
            raise ValueError("endpoint_service must be provided")
        
        self._endpoint_service = endpoint_service
        
        super().__init__(**kwargs)
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt."""
        return self._endpoint_service.generate_response(prompt, **kwargs)
    
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
        
        # Initialize Bedrock endpoint services (only endpoint approach)
        endpoint_url = getattr(settings, 'bedrock_endpoint_url', None)
        if not endpoint_url:
            raise ValueError("Bedrock endpoint URL not configured - only endpoint approach is supported")
        
        from .bedrock_endpoint_service import BedrockEndpointService
        endpoint_service = BedrockEndpointService(endpoint_url)
        self.bedrock_endpoint_embedding = BedrockEndpointEmbeddingService(endpoint_service)
        self.bedrock_endpoint_llm = BedrockEndpointLLMWrapper(endpoint_service)
        
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
                opensearch_client_kwargs['verify_certs'] = settings.opensearch.verify_certs
                
                if not settings.opensearch.verify_certs:
                    opensearch_client_kwargs['ssl_show_warn'] = False
                    opensearch_client_kwargs['ssl_assert_hostname'] = False
                    
                    # Add SSL context configuration to handle SSL handshake issues
                    import ssl
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    # Configure SSL context for OpenSearch compatibility
                    try:
                        ssl_context.set_ciphers('HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')
                    except ssl.SSLError:
                        # If the cipher string fails, try a more basic one
                        ssl_context.set_ciphers('DEFAULT')
                    
                    opensearch_client_kwargs['ssl_context'] = ssl_context
            
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
            # Use endpoint service for embeddings (only supported approach)
            embed_model = CustomBedrockEmbedding(endpoint_service=self.bedrock_endpoint_embedding)
            logger.info("Using custom embedding wrapper for Bedrock endpoint")
            
            # Use endpoint service for LLM (only supported approach)
            llm = CustomBedrockLLM(endpoint_service=self.bedrock_endpoint_llm)
            logger.info("Using custom LLM wrapper for Bedrock endpoint")
            
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
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        document_type: str
    ) -> bool:
        """Add document to the index with JSON to Dolphin format conversion."""
        try:
            # Create individual documents for each model/view/relationship entity
            documents = []
            if self.content_processor.is_json_content(content):
                from ..models.simple_models import DocumentType as DocType
                # Map document type string to enum
                # FORCE CORRECT TYPE FOR LOOKUP FILES
                if "lookup" in document_id.lower() or "lookups" in document_id.lower():
                    doc_type_enum = DocType.LOOKUP_METADATA
                    logger.info("FORCED document type to LOOKUP_METADATA based on document_id", document_id=document_id)
                elif document_type.lower() == "schema":
                    doc_type_enum = DocType.SCHEMA
                elif document_type.lower() == "lookup_metadata":
                    doc_type_enum = DocType.LOOKUP_METADATA
                else:
                    doc_type_enum = DocType.REPORT
                
                # Create individual documents using the new approach
                logger.info(
                    "About to call create_individual_documents",
                    document_id=document_id,
                    document_type=document_type,
                    doc_type_enum=doc_type_enum.value if hasattr(doc_type_enum, 'value') else str(doc_type_enum),
                    content_length=len(content),
                    content_preview=content[:200],
                    is_json=self.content_processor.is_json_content(content)
                )
                
                # Use hierarchical document processing for new types
                if doc_type_enum in [DocType.DDL, DocType.BUSINESS_DESC, DocType.BUSINESS_RULES, 
                                   DocType.COLUMN_DETAILS, DocType.LOOKUP_METADATA]:
                    individual_documents = self.content_processor.create_hierarchical_documents(content, doc_type_enum)
                else:
                    # Fall back to legacy processing for old schema files
                    individual_documents = self.content_processor.create_individual_documents_LEGACY(content, doc_type_enum)
                
                logger.info(
                    "Created individual documents for separate storage",
                    document_id=document_id,
                    num_documents=len(individual_documents),
                    entity_types=[doc["metadata"].get("entity_type") for doc in individual_documents],
                    chunk_types=[doc["metadata"].get("chunk_type") for doc in individual_documents]
                )
                
                # Convert each individual document to a complete LlamaIndex document
                for i, doc_data in enumerate(individual_documents):
                    # Create unique entity ID based on entity type and name
                    entity_type = doc_data["metadata"].get("entity_type", "entity")
                    entity_name = (doc_data["metadata"].get("table_name") or 
                                 doc_data["metadata"].get("view_name") or 
                                 doc_data["metadata"].get("relationship_name") or
                                 doc_data["metadata"].get("lookup_name") or 
                                 f"entity_{i}")
                    entity_id = f"{document_id}_{entity_type}_{entity_name}"
                    
                    doc_metadata = {
                        "document_id": entity_id,  # Each entity gets its own document ID
                        "parent_document_id": document_id,  # Track original document
                        "document_type": document_type,
                        "entity_index": i,
                        "total_entities": len(individual_documents),
                        "is_individual_entity": True,
                        **metadata,
                        **doc_data["metadata"]  # Include all entity metadata
                    }
                    
                    document = LlamaDocument(
                        text=doc_data["content"],
                        metadata=doc_metadata,
                        id_=entity_id
                    )
                    documents.append(document)
            else:
                # Fallback to traditional chunking for non-JSON content
                processed_content = content
                doc_metadata = {
                    "document_id": document_id,
                    "document_type": document_type,
                    "is_json_converted": False,
                    **metadata
                }
                
                document = LlamaDocument(
                    text=processed_content,
                    metadata=doc_metadata,
                    id_=f"doc_{document_id}"
                )
                documents.append(document)
            
            # Process all documents (no further chunking for individual entities)
            nodes = []
            for doc in documents:
                # For individual entities, store as single nodes (no further chunking)
                if doc.metadata.get("is_individual_entity", False):
                    # Create a single node from the complete entity document
                    from llama_index.core.schema import TextNode
                    node = TextNode(
                        text=doc.text,
                        metadata=doc.metadata,
                        id_=doc.id_
                    )
                    nodes.append(node)
                else:
                    # Use traditional chunking for non-individual documents
                    splitter = SentenceSplitter(
                        chunk_size=settings.app.chunk_size,
                        chunk_overlap=settings.app.chunk_overlap
                    )
                    doc_nodes = splitter.get_nodes_from_documents([doc])
                    nodes.extend(doc_nodes)
            
            # Validate nodes before processing
            if not nodes:
                logger.error("No nodes created from documents", document_id=document_id)
                return False
            
            # Filter out empty nodes and validate content
            valid_nodes = []
            for i, node in enumerate(nodes):
                if not node.text or len(node.text.strip()) < 10:
                    logger.warning("Skipping empty or too short node", 
                                 document_id=document_id, 
                                 node_index=i,
                                 text_length=len(node.text) if node.text else 0)
                    continue
                
                node.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(nodes)
                })
                valid_nodes.append(node)
            
            if not valid_nodes:
                logger.error("No valid nodes after filtering", document_id=document_id)
                return False
            
            nodes = valid_nodes
            logger.info("Processed nodes", document_id=document_id, total_nodes=len(nodes), valid_nodes=len(valid_nodes))
            
            # Insert nodes into index
            try:
                logger.info(
                    "Attempting to insert nodes",
                    document_id=document_id,
                    num_nodes=len(nodes),
                    first_node_metadata=nodes[0].metadata if nodes else None
                )
                
                # Pre-validate that nodes can generate embeddings
                for node in nodes:
                    if not node.text:
                        raise ValueError(f"Node has empty text: {node.node_id}")
                
                self.index.insert_nodes(nodes)
                
                logger.info(
                    "Successfully inserted nodes into index",
                    document_id=document_id,
                    num_chunks=len(nodes),
                    document_type=document_type
                )
                
                # Verify the insertion worked by searching for the document
                verification_results = self.search_similar(
                    query="document",
                    document_ids=[document_id],
                    similarity_top_k=5
                )
                
                logger.info(
                    "Verification search results",
                    document_id=document_id,
                    verification_count=len(verification_results)
                )
                
                return True
                
            except Exception as insert_error:
                logger.error(
                    "Failed to insert nodes into index",
                    document_id=document_id,
                    error=str(insert_error),
                    error_type=type(insert_error).__name__
                )
                return False
            
        except Exception as e:
            logger.error("Failed to add document to index", document_id=document_id, error=str(e))
            return False
    
    def search_similar(
        self,
        query: str,
        retriever_type: str = "hybrid",
        similarity_top_k: Optional[int] = None,
        document_type: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content using specified retriever with semantic awareness."""
        try:
            logger.info("Starting semantic-aware search", 
                       query=query,
                       retriever_type=retriever_type,
                       similarity_top_k=similarity_top_k,
                       document_type=document_type,
                       document_ids=document_ids)
            
            # Select retriever
            retriever = self.retrievers.get(retriever_type, self.retrievers["vector"])
            
            # Update similarity_top_k if provided - use higher number for semantic filtering
            search_top_k = similarity_top_k if similarity_top_k else 20  # Get more candidates for semantic filtering
            retriever.similarity_top_k = search_top_k
            
            # Perform retrieval
            nodes = retriever.retrieve(query)
            
            logger.info("Retrieved nodes from vector search",
                       raw_node_count=len(nodes),
                       query_terms=self._extract_query_terms(query))
            
            # Format results with semantic ranking and enhanced filtering
            results = []
            query_terms = self._extract_query_terms(query)
            
            for node in nodes:
                node_data = {
                    "id": node.node_id,
                    "score": getattr(node, 'score', 1.0),
                    "content": node.text,
                    "metadata": node.metadata,
                    "document_id": node.metadata.get("document_id"),
                    "document_type": node.metadata.get("document_type")
                }
                
                # Apply basic filters first
                should_include = True
                
                if document_ids:
                    node_doc_id = node.metadata.get("document_id")
                    logger.debug(f"Document filtering: looking for {document_ids}, found node with doc_id='{node_doc_id}', match={node_doc_id in document_ids}")
                    if node_doc_id not in document_ids:
                        should_include = False
                
                if document_type and should_include:
                    node_doc_type = node.metadata.get("document_type")
                    if node_doc_type != document_type:
                        should_include = False
                
                if should_include:
                    # Apply semantic ranking based on business context
                    semantic_score = self._calculate_semantic_relevance(node.metadata, query_terms, query)
                    node_data["semantic_score"] = semantic_score
                    node_data["combined_score"] = (node_data["score"] * 0.7) + (semantic_score * 0.3)
                    
                    logger.debug("Node evaluation",
                               node_id=node.node_id,
                               chunk_type=node.metadata.get("chunk_type"),
                               entity_type=node.metadata.get("entity_type"),
                               business_terms=node.metadata.get("business_terms"),
                               vector_score=node_data["score"],
                               semantic_score=semantic_score,
                               combined_score=node_data["combined_score"])
                    
                    results.append(node_data)
            
            # Sort by combined score (vector similarity + semantic relevance)
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            # Limit to requested number of results
            final_limit = similarity_top_k if similarity_top_k else 5
            results = results[:final_limit]
            
            logger.info("Completed semantic-aware search",
                       query_length=len(query),
                       raw_results=len(nodes),
                       filtered_results=len(results),
                       retriever_type=retriever_type,
                       top_chunk_types=[r["metadata"].get("chunk_type") for r in results[:3]],
                       top_scores=[f"{r['combined_score']:.3f}" for r in results[:3]])
            
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
                        "content_snippet": node.text,  # No truncation for debugging
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
    
    def delete_document(self, document_id: str) -> bool:
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
        document_id: str,
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
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Build metadata filters for retrieval."""
        filters = {}
        
        if document_type:
            filters["document_type"] = document_type
        
        if document_ids:
            filters["document_id"] = document_ids  # Already strings
        
        return filters
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """Get information about a document's vectors."""
        try:
            logger.info("Getting document info for", document_id=document_id, document_id_type=type(document_id))
            
            # Try to search for documents with this ID, but handle search failures gracefully
            results = []
            try:
                results = self.search_similar(
                    query="document",  # Simple query
                    document_ids=[document_id],
                    similarity_top_k=100
                )
                logger.info("Search results for document", document_id=document_id, num_results=len(results))
            except Exception as search_error:
                logger.warning("Search failed, document likely doesn't exist", 
                             document_id=document_id, 
                             error=str(search_error))
                # Return not found immediately if search fails
                return {
                    "document_id": document_id,
                    "num_chunks": 0,
                    "metadata": {},
                    "status": "not_found",
                    "error": "Search failed - document likely doesn't exist"
                }
            
            if not results:
                logger.warning("No chunks found for document", document_id=document_id)
                return {
                    "document_id": document_id,
                    "num_chunks": 0,
                    "metadata": {},
                    "status": "not_found"
                }
            
            info = {
                "document_id": document_id,
                "num_chunks": len(results),
                "metadata": results[0]["metadata"],
                "status": "found"
            }
            
            return info
            
        except Exception as e:
            logger.error("Failed to get document info", document_id=document_id, error=str(e))
            return {
                "document_id": document_id, 
                "num_chunks": 0, 
                "metadata": {},
                "status": "error",
                "error": str(e)
            }
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the index for debugging."""
        try:
            # Get retriever and perform raw search
            retriever = self.retrievers.get("vector")
            retriever.similarity_top_k = 1000
            
            # Search for all documents
            nodes = retriever.retrieve("document")  # Generic query
            logger.info(f"Raw retriever nodes count: {len(nodes)}")
            
            # Group by document_id and show unique document_ids
            docs_by_id = {}
            unique_doc_ids = set()
            
            for i, node in enumerate(nodes):
                doc_id = node.metadata.get("document_id")
                doc_type = node.metadata.get("document_type")
                unique_doc_ids.add(doc_id)
                
                if i < 10:  # Log first 10 results for debugging
                    logger.info(f"Node {i}: doc_id={doc_id}, type={doc_type}, node_id={node.node_id}")
                
                if doc_id not in docs_by_id:
                    docs_by_id[doc_id] = {
                        "document_id": doc_id,
                        "document_type": doc_type,
                        "chunk_count": 0,
                        "sample_metadata": node.metadata
                    }
                docs_by_id[doc_id]["chunk_count"] += 1
            
            logger.info(f"Unique document IDs found: {list(unique_doc_ids)}")
            logger.info(f"Document counts: {[(k, v['chunk_count']) for k, v in docs_by_id.items()]}")
            
            return list(docs_by_id.values())
            
        except Exception as e:
            logger.error("Failed to list documents", error=str(e))
            return []
    
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
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from the search query."""
        import re
        # Simple term extraction - can be enhanced with NLP
        terms = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'all', 'with', 'that', 'what', 'have', 'from', 'they', 'this'}
        meaningful_terms = [term for term in terms if term not in stop_words]
        
        return meaningful_terms
    
    def _calculate_semantic_relevance(self, node_metadata: Dict[str, Any], query_terms: List[str], full_query: str) -> float:
        """Calculate semantic relevance score based on business context and metadata."""
        score = 0.0
        
        # Extract relevant metadata fields
        chunk_type = node_metadata.get("chunk_type", "")
        entity_type = node_metadata.get("entity_type", "")
        business_terms = node_metadata.get("business_terms", [])
        table_name = node_metadata.get("table_name", "")
        view_name = node_metadata.get("view_name", "")
        business_domain = node_metadata.get("business_domain", "")
        
        # Boost score for matching business terms
        if business_terms:
            for query_term in query_terms:
                if query_term in business_terms:
                    score += 0.3
        
        # Boost score for relevant chunk types based on query intent
        if self._is_schema_query(full_query):
            if chunk_type in ["table_entity", "view_entity", "schema_overview"]:
                score += 0.4
            if chunk_type == "relationship_domain":
                score += 0.2
        
        # Boost score for financial domain queries
        financial_terms = ["deal", "deals", "tranche", "tranches", "fixed", "income", "announced", "status", "bond", "security"]
        query_has_financial_terms = any(term in full_query.lower() for term in financial_terms)
        
        if query_has_financial_terms:
            if business_domain == "financial_instruments":
                score += 0.5
            if entity_type in ["deal_entity", "tranche_entity", "security_entity"]:
                score += 0.4
            if any(financial_term in table_name.lower() for financial_term in financial_terms):
                score += 0.3
        
        # Boost score for exact table/entity name matches
        for query_term in query_terms:
            if query_term in table_name.lower() or query_term in view_name.lower():
                score += 0.6
        
        # Boost score for status-related queries
        status_terms = ["status", "state", "announced", "pending", "active", "completed"]
        if any(term in full_query.lower() for term in status_terms):
            if entity_type == "status_entity" or "status" in table_name.lower():
                score += 0.4
        
        # Normalize score to 0-1 range
        return min(score, 1.0)
    
    def _is_schema_query(self, query: str) -> bool:
        """Determine if the query is asking about schema/structure information."""
        schema_indicators = ["table", "column", "field", "structure", "schema", "database", "what tables", "which tables"]
        return any(indicator in query.lower() for indicator in schema_indicators)
    
    def two_step_metadata_retrieval_LEGACY(
        self,
        query: str,
        similarity_top_k: int = 10,
        document_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """LEGACY: Perform 2-step retrieval: first models/views, then relationships, then pruning.
        
        WARNING: This method is deprecated and replaced by HierarchicalContextService.
        
        This method implements the user's requested approach:
        1. Semantic search for relevant models/views with rewritten prompts
        2. Search for relationships relevant to the returned models/views
        3. Pruning step to remove unneeded metadata
        
        Args:
            query: The user's natural language query
            similarity_top_k: Maximum number of results per step
            document_type: Filter by document type if needed
            
        Returns:
            Dictionary containing retrieved and pruned metadata
        """
        try:
            logger.info("=== STARTING 2-STEP METADATA RETRIEVAL ===")
            logger.info("Step 1: Retrieving relevant models and views", 
                       query=query, similarity_top_k=similarity_top_k)
            
            # Debug: First check what documents exist in the vector store
            all_docs = self.list_all_documents()
            logger.info("Available documents in vector store", 
                       total_docs=len(all_docs),
                       doc_types=[doc.get("document_type") for doc in all_docs],
                       sample_doc_ids=[doc.get("document_id") for doc in all_docs[:5]])
            
            # Step 1: Get relevant models and views with optimized prompt rewriting
            models_views_results = self._step1_retrieve_models_views(query, similarity_top_k)
            
            logger.info("Step 1 completed", 
                       num_models_views=len(models_views_results),
                       entities=[r["metadata"].get("table_name") or r["metadata"].get("view_name") 
                               for r in models_views_results[:5]])
            
            # Step 2: Get relationships relevant to the retrieved models/views
            logger.info("Step 2: Retrieving relevant relationships")
            relationships_results = self._step2_retrieve_relationships(models_views_results, query, similarity_top_k)
            
            logger.info("Step 2 completed", 
                       num_relationships=len(relationships_results),
                       relationship_names=[r["metadata"].get("relationship_name") 
                                         for r in relationships_results[:5]])
            
            # Step 3: Pruning step to remove unneeded metadata
            logger.info("Step 3: Pruning metadata")
            pruned_metadata = self._step3_prune_metadata(models_views_results, relationships_results, query)
            
            logger.info("=== 2-STEP RETRIEVAL COMPLETE ===", 
                       final_models=len(pruned_metadata["models"]),
                       final_views=len(pruned_metadata["views"]),
                       final_relationships=len(pruned_metadata["relationships"]))
            
            return pruned_metadata
            
        except Exception as e:
            logger.error("Failed in 2-step metadata retrieval", error=str(e))
            return {
                "models": [],
                "views": [],
                "relationships": [],
                "error": str(e)
            }
    
    def _step1_retrieve_models_views(self, query: str, similarity_top_k: int) -> List[Dict[str, Any]]:
        """Step 1: Retrieve relevant models and views with optimized prompt rewriting."""
        try:
            # Create optimized search queries for models and views
            rewritten_queries = self._rewrite_query_for_models_views(query)
            logger.info("Step 1: Rewritten queries for models/views", 
                       original_query=query,
                       rewritten_queries=rewritten_queries)
            
            all_results = []
            
            for i, rewritten_query in enumerate(rewritten_queries):
                logger.info(f"Step 1.{i+1}: Searching for models/views", query=rewritten_query)
                
                # Search specifically for individual models and views
                results = self.search_similar(
                    query=rewritten_query,
                    retriever_type="hybrid",
                    similarity_top_k=similarity_top_k,
                    document_type="schema"
                )
                
                logger.info(f"Step 1.{i+1}: Raw search results", 
                           num_results=len(results),
                           result_types=[r["metadata"].get("entity_type") for r in results],
                           result_names=[r["metadata"].get("table_name") or r["metadata"].get("view_name") 
                                       for r in results])
                
                # Filter to only models and views
                filtered_results = [
                    r for r in results 
                    if r["metadata"].get("entity_type") in ["model", "view"]
                ]
                
                logger.info(f"Step 1.{i+1}: Filtered results (models/views only)", 
                           num_filtered=len(filtered_results),
                           filtered_names=[r["metadata"].get("table_name") or r["metadata"].get("view_name") 
                                         for r in filtered_results])
                
                all_results.extend(filtered_results)
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_and_rank_results(all_results, max_results=similarity_top_k)
            
            return unique_results
            
        except Exception as e:
            logger.error("Failed in step 1 retrieval", error=str(e))
            return []
    
    def _step2_retrieve_relationships(self, models_views_results: List[Dict[str, Any]], query: str, similarity_top_k: int) -> List[Dict[str, Any]]:
        """Step 2: Retrieve relationships relevant to the models/views from step 1."""
        try:
            # Extract table/view names from step 1 results
            entity_names = set()
            for result in models_views_results:
                metadata = result["metadata"]
                if "table_name" in metadata:
                    entity_names.add(metadata["table_name"])
                if "view_name" in metadata:
                    entity_names.add(metadata["view_name"])
            
            if not entity_names:
                logger.warning("No entity names found from step 1 results")
                return []
            
            logger.info("Searching for relationships involving entities", entities=list(entity_names))
            
            # Create relationship-focused search queries
            relationship_queries = self._create_relationship_queries(entity_names, query)
            
            all_relationship_results = []
            
            for rel_query in relationship_queries:
                logger.info("Searching for relationships", query=rel_query)
                
                # Search for relationships
                results = self.search_similar(
                    query=rel_query,
                    retriever_type="hybrid",
                    similarity_top_k=similarity_top_k,
                    document_type="schema"
                )
                
                # Filter to only relationships that involve our entities
                relevant_relationships = []
                for r in results:
                    if r["metadata"].get("entity_type") == "relationship":
                        relationship_tables = r["metadata"].get("tables", [])
                        # Check if this relationship involves any of our entities
                        if any(table in entity_names for table in relationship_tables):
                            relevant_relationships.append(r)
                
                all_relationship_results.extend(relevant_relationships)
            
            # Deduplicate and rank relationship results
            unique_relationships = self._deduplicate_and_rank_results(all_relationship_results, max_results=similarity_top_k)
            
            return unique_relationships
            
        except Exception as e:
            logger.error("Failed in step 2 retrieval", error=str(e))
            return []
    
    def _step3_prune_metadata(self, models_views: List[Dict[str, Any]], relationships: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Step 3: Prune metadata to remove unneeded information."""
        try:
            # Analyze query to determine what information is actually needed
            query_analysis = self._analyze_query_requirements(query)
            
            pruned_models = []
            pruned_views = []
            pruned_relationships = []
            
            # Prune models based on relevance and query requirements
            for result in models_views:
                metadata = result["metadata"]
                if metadata.get("entity_type") == "model":
                    pruned_model = self._prune_model_metadata(result, query_analysis)
                    if pruned_model:
                        pruned_models.append(pruned_model)
                elif metadata.get("entity_type") == "view":
                    pruned_view = self._prune_view_metadata(result, query_analysis)
                    if pruned_view:
                        pruned_views.append(pruned_view)
            
            # Prune relationships to only include those relevant to remaining models/views
            retained_entity_names = set()
            for model in pruned_models:
                retained_entity_names.add(model["metadata"].get("table_name", ""))
            for view in pruned_views:
                retained_entity_names.add(view["metadata"].get("view_name", ""))
            
            for relationship in relationships:
                relationship_tables = relationship["metadata"].get("tables", [])
                # Keep relationship if it connects retained entities
                if any(table in retained_entity_names for table in relationship_tables):
                    pruned_relationship = self._prune_relationship_metadata(relationship, query_analysis)
                    if pruned_relationship:
                        pruned_relationships.append(pruned_relationship)
            
            logger.info("Pruning completed", 
                       original_models_views=len(models_views),
                       original_relationships=len(relationships),
                       pruned_models=len(pruned_models),
                       pruned_views=len(pruned_views),
                       pruned_relationships=len(pruned_relationships))
            
            return {
                "models": pruned_models,
                "views": pruned_views,
                "relationships": pruned_relationships,
                "query_analysis": query_analysis
            }
            
        except Exception as e:
            logger.error("Failed in step 3 pruning", error=str(e))
            return {
                "models": models_views,  # Return unpruned as fallback
                "views": [],
                "relationships": relationships,
                "query_analysis": {},
                "error": str(e)
            }
    
    def _rewrite_query_for_models_views(self, query: str) -> List[str]:
        """Create optimized search queries for finding relevant models and views."""
        queries = [query]  # Always include original
        query_lower = query.lower()
        
        # Extract entities and create focused queries
        if any(term in query_lower for term in ['user', 'users', 'customer', 'customers']):
            queries.extend([
                "users table customers user information",
                "user entity customer data table",
                "users customers database table model"
            ])
        
        if any(term in query_lower for term in ['order', 'orders', 'purchase', 'transaction']):
            queries.extend([
                "orders table transactions purchases",
                "order entity transaction data",
                "orders purchases database table model"
            ])
        
        if any(term in query_lower for term in ['product', 'products', 'item', 'items']):
            queries.extend([
                "products table items inventory",
                "product entity item data",
                "products items database table model"
            ])
        
        # Add business domain queries
        if any(term in query_lower for term in ['sales', 'revenue', 'total', 'amount']):
            queries.extend([
                "sales revenue tables financial data",
                "sales summary view analytical data",
                "revenue amount tables financial model"
            ])
        
        # Add Fixed Income domain queries
        if any(term in query_lower for term in ['deal', 'deals', 'fixed income', 'tranche', 'tranches']):
            queries.extend([
                "deal deals table fixed income financial",
                "tranche tranches table deal data",
                "fixed income deals database table model",
                "termsheet deals tranche information"
            ])
        
        if any(term in query_lower for term in ['announced', 'status', 'state']):
            queries.extend([
                "status announced table deal information",
                "status state table database model",
                "announced status deals tranche"
            ])
        
        if any(term in query_lower for term in ['investor', 'trades', 'trading']):
            queries.extend([
                "investor trades table trading data",
                "investor trading database table model",
                "trades investor information table"
            ])
        
        if any(term in query_lower for term in ['termsheet', 'term sheet']):
            queries.extend([
                "termsheet term sheet view table",
                "termsheet deals information database",
                "v_termsheet view termsheet data"
            ])
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))  # Preserves order
        return unique_queries[:6]  # Increased limit to handle more domain-specific queries
    
    def _create_relationship_queries(self, entity_names: set, original_query: str) -> List[str]:
        """Create queries to find relationships involving specific entities."""
        queries = []
        
        entity_list = list(entity_names)
        
        # Create queries that combine entity names with relationship terms
        if len(entity_list) >= 2:
            queries.append(f"relationship join {entity_list[0]} {entity_list[1]}")
            queries.append(f"{entity_list[0]} {entity_list[1]} connection relationship")
        
        # Add general relationship queries for each entity
        for entity in entity_list:
            queries.extend([
                f"{entity} relationships joins connections",
                f"relationship {entity} table connections",
                f"{entity} foreign key relationships"
            ])
        
        # Add original query context
        queries.append(f"relationships {original_query}")
        
        return queries[:6]  # Limit to 6 relationship queries
    
    def _analyze_query_requirements(self, query: str) -> Dict[str, Any]:
        """Analyze the query to understand what metadata is actually needed."""
        query_lower = query.lower()
        
        analysis = {
            "needs_aggregation": any(term in query_lower for term in ['count', 'sum', 'total', 'average', 'max', 'min']),
            "needs_filtering": any(term in query_lower for term in ['where', 'filter', 'specific', 'only', 'particular']),
            "needs_sorting": any(term in query_lower for term in ['order', 'sort', 'top', 'highest', 'lowest', 'recent']),
            "needs_joins": any(term in query_lower for term in ['join', 'relate', 'connect', 'with', 'together']),
            "key_terms": self._extract_query_terms(query),
            "focus_areas": []
        }
        
        # Identify focus areas
        if any(term in query_lower for term in ['user', 'customer', 'person']):
            analysis["focus_areas"].append("user_data")
        if any(term in query_lower for term in ['order', 'purchase', 'transaction']):
            analysis["focus_areas"].append("transaction_data")
        if any(term in query_lower for term in ['product', 'item', 'inventory']):
            analysis["focus_areas"].append("product_data")
        
        return analysis
    
    def _prune_model_metadata(self, model_result: Dict[str, Any], query_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prune model metadata to keep only relevant information."""
        try:
            metadata = model_result["metadata"]
            table_name = metadata.get("table_name", "")
            columns = metadata.get("columns", [])
            
            # Always keep core metadata
            pruned_result = {
                "content": model_result["content"],
                "metadata": {
                    "entity_type": "model",
                    "table_name": table_name,
                    "catalog": metadata.get("catalog", ""),
                    "schema": metadata.get("schema", ""),
                    "business_terms": metadata.get("business_terms", []),
                    "classified_entity_type": metadata.get("classified_entity_type", ""),
                    "columns": columns,  # Keep all columns for now, could be pruned further if needed
                    "pruned": True
                },
                "score": model_result.get("score", 0.0),
                "combined_score": model_result.get("combined_score", model_result.get("score", 0.0))
            }
            
            # Check relevance based on query analysis
            table_relevant = False
            key_terms = query_analysis.get("key_terms", [])
            
            # Check if table name matches query terms
            if any(term in table_name.lower() for term in key_terms):
                table_relevant = True
            
            # Check if business terms match
            business_terms = metadata.get("business_terms", [])
            if any(term in business_terms for term in key_terms):
                table_relevant = True
            
            # Check focus areas
            focus_areas = query_analysis.get("focus_areas", [])
            if focus_areas:
                if "user_data" in focus_areas and any(term in table_name.lower() for term in ['user', 'customer', 'person']):
                    table_relevant = True
                if "transaction_data" in focus_areas and any(term in table_name.lower() for term in ['order', 'transaction', 'purchase']):
                    table_relevant = True
                if "product_data" in focus_areas and any(term in table_name.lower() for term in ['product', 'item', 'inventory']):
                    table_relevant = True
            
            return pruned_result if table_relevant else None
            
        except Exception as e:
            logger.error("Failed to prune model metadata", error=str(e))
            return model_result  # Return original as fallback
    
    def _prune_view_metadata(self, view_result: Dict[str, Any], query_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prune view metadata to keep only relevant information."""
        try:
            metadata = view_result["metadata"]
            view_name = metadata.get("view_name", "")
            
            pruned_result = {
                "content": view_result["content"],
                "metadata": {
                    "entity_type": "view",
                    "view_name": view_name,
                    "catalog": metadata.get("catalog", ""),
                    "schema": metadata.get("schema", ""),
                    "business_terms": metadata.get("business_terms", []),
                    "columns": metadata.get("columns", []),
                    "pruned": True
                },
                "score": view_result.get("score", 0.0),
                "combined_score": view_result.get("combined_score", view_result.get("score", 0.0))
            }
            
            # Similar relevance check as models
            view_relevant = False
            key_terms = query_analysis.get("key_terms", [])
            
            if any(term in view_name.lower() for term in key_terms):
                view_relevant = True
            
            business_terms = metadata.get("business_terms", [])
            if any(term in business_terms for term in key_terms):
                view_relevant = True
            
            return pruned_result if view_relevant else None
            
        except Exception as e:
            logger.error("Failed to prune view metadata", error=str(e))
            return view_result  # Return original as fallback
    
    def _prune_relationship_metadata(self, relationship_result: Dict[str, Any], query_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prune relationship metadata to keep only relevant information."""
        try:
            metadata = relationship_result["metadata"]
            
            pruned_result = {
                "content": relationship_result["content"],
                "metadata": {
                    "entity_type": "relationship",
                    "relationship_name": metadata.get("relationship_name", ""),
                    "catalog": metadata.get("catalog", ""),
                    "schema": metadata.get("schema", ""),
                    "tables": metadata.get("tables", []),
                    "relationship_type": metadata.get("relationship_type", ""),
                    "business_terms": metadata.get("business_terms", []),
                    "pruned": True
                },
                "score": relationship_result.get("score", 0.0),
                "combined_score": relationship_result.get("combined_score", relationship_result.get("score", 0.0))
            }
            
            # Keep all relationships that made it this far as they were already filtered
            # by relevance to the retained models/views
            return pruned_result
            
        except Exception as e:
            logger.error("Failed to prune relationship metadata", error=str(e))
            return relationship_result  # Return original as fallback
    
    def _deduplicate_and_rank_results(self, results: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
        """Deduplicate search results and rank by combined score."""
        seen_ids = set()
        unique_results = []
        
        for result in results:
            result_id = result.get("id")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by combined score if available, otherwise by regular score
        unique_results.sort(key=lambda x: x.get("combined_score", x.get("score", 0.0)), reverse=True)
        
        return unique_results[:max_results]