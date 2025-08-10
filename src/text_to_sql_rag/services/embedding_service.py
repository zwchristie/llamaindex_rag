"""
Embedding and retrieval service using Bedrock API Gateway.
"""

import json
import httpx
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import asyncio

from ..models.view_models import ViewMetadata, ViewEmbedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Bedrock API Gateway."""
    
    def __init__(self, endpoint_url: str, embedding_model: str, use_mock: bool = False):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.embedding_model = embedding_model
        self.embedding_dimension = None  # Will be detected dynamically
        self.use_mock = use_mock
        
        if self.use_mock:
            self.embedding_dimension = 1536  # Standard dimension for mock embeddings
            logger.info("Using mock embeddings for demo purposes")
        
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding based on text hash."""
        # Create deterministic embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash to floats and normalize
        embedding = []
        for i in range(self.embedding_dimension):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Scale to [-1, 1]
            embedding.append(value)
        
        # Normalize the vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Bedrock API Gateway or mock."""
        if self.use_mock:
            return self._generate_mock_embedding(text)
            
        try:
            payload = {
                "model_id": self.embedding_model,
                "invoke_type": "embedding",
                "query": text
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Handle different response formats
                if "embedding" in result:
                    embedding = result["embedding"]
                elif "body" in result:
                    body_data = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
                    embedding = body_data.get("embedding", [])
                else:
                    logger.error(f"Unexpected response format: {result}")
                    raise ValueError("Could not extract embedding from response")
                
                # Detect embedding dimension on first call
                if self.embedding_dimension is None:
                    self.embedding_dimension = len(embedding)
                    logger.info(f"Detected embedding dimension: {self.embedding_dimension}")
                
                return embedding
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error generating embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Process batch concurrently
            tasks = [self.get_embedding(text) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error embedding text {i+j}: {result}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * (self.embedding_dimension or 1536))
                else:
                    batch_embeddings.append(result)
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return embeddings
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the embedding dimension (detected after first embedding call)."""
        return self.embedding_dimension


class VectorService:
    """Service for vector storage and retrieval using OpenSearch."""
    
    def __init__(self, opensearch_client, index_name: str, vector_field: str):
        self.client = opensearch_client
        self.index_name = index_name
        self.vector_field = vector_field
        self.text_field = "full_text"
        self.metadata_field = "metadata"
    
    async def create_index(self, embedding_dimension: int):
        """Create OpenSearch index with vector mapping."""
        try:
            # Check if index exists
            if await self._index_exists():
                logger.info(f"Index {self.index_name} already exists")
                return
            
            mapping = {
                "mappings": {
                    "properties": {
                        "view_name": {
                            "type": "keyword"
                        },
                        self.text_field: {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        self.vector_field: {
                            "type": "knn_vector",
                            "dimension": embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        self.metadata_field: {
                            "type": "object",
                            "enabled": True
                        },
                        "created_at": {
                            "type": "date"
                        }
                    }
                },
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 512
                    }
                }
            }
            
            response = await self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            
            logger.info(f"Created OpenSearch index {self.index_name} with dimension {embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    async def _index_exists(self) -> bool:
        """Check if index exists."""
        try:
            return await self.client.indices.exists(index=self.index_name)
        except:
            return False
    
    async def index_view_embedding(self, view_embedding: ViewEmbedding):
        """Index a view embedding document."""
        try:
            doc = {
                "view_name": view_embedding.view_name,
                self.text_field: view_embedding.full_text,
                self.vector_field: view_embedding.embedding,
                self.metadata_field: view_embedding.metadata.dict(),
                "created_at": view_embedding.created_at.isoformat()
            }
            
            await self.client.index(
                index=self.index_name,
                id=view_embedding.view_name,  # Use view_name as document ID
                body=doc
            )
            
            logger.debug(f"Indexed embedding for view {view_embedding.view_name}")
            
        except Exception as e:
            logger.error(f"Error indexing view {view_embedding.view_name}: {e}")
            raise
    
    async def search_similar_views(self, query_embedding: List[float], k: int = 5) -> List[Tuple[ViewMetadata, float]]:
        """Search for similar views using vector similarity."""
        try:
            search_body = {
                "size": k,
                "query": {
                    "knn": {
                        self.vector_field: {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                },
                "_source": [self.metadata_field, "view_name"]
            }
            
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                metadata_dict = hit["_source"][self.metadata_field]
                view_metadata = ViewMetadata(**metadata_dict)
                results.append((view_metadata, score))
            
            logger.info(f"Vector search returned {len(results)} similar views")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise
    
    async def hybrid_search(self, query_text: str, query_embedding: List[float], k: int = 5, text_weight: float = 0.3, vector_weight: float = 0.7) -> List[Tuple[ViewMetadata, float]]:
        """Perform hybrid search combining text and vector similarity."""
        try:
            search_body = {
                "size": k,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    self.text_field: {
                                        "query": query_text,
                                        "boost": text_weight
                                    }
                                }
                            },
                            {
                                "knn": {
                                    self.vector_field: {
                                        "vector": query_embedding,
                                        "k": k,
                                        "boost": vector_weight
                                    }
                                }
                            }
                        ]
                    }
                },
                "_source": [self.metadata_field, "view_name"]
            }
            
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                score = hit["_score"]
                metadata_dict = hit["_source"][self.metadata_field]
                view_metadata = ViewMetadata(**metadata_dict)
                results.append((view_metadata, score))
            
            logger.info(f"Hybrid search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search only
            return await self.search_similar_views(query_embedding, k)
    
    async def delete_view_embedding(self, view_name: str):
        """Delete a view embedding document."""
        try:
            await self.client.delete(
                index=self.index_name,
                id=view_name
            )
            logger.info(f"Deleted embedding for view {view_name}")
            
        except Exception as e:
            if "not_found" not in str(e).lower():
                logger.error(f"Error deleting view embedding {view_name}: {e}")
                raise
    
    async def reindex_all_views(self, views: List[ViewMetadata], embedding_service: EmbeddingService):
        """Reindex all views with fresh embeddings."""
        try:
            logger.info(f"Starting reindex of {len(views)} views")
            
            # Delete existing index
            if await self._index_exists():
                await self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index {self.index_name}")
            
            # Generate embeddings for all views
            texts = [view.generate_full_text() for view in views]
            embeddings = await embedding_service.get_embeddings_batch(texts)
            
            # Create new index with detected dimension
            await self.create_index(embedding_service.get_embedding_dimension())
            
            # Index all view embeddings
            for view, embedding in zip(views, embeddings):
                view_embedding = ViewEmbedding(
                    view_name=view.view_name,
                    full_text=view.generate_full_text(),
                    embedding=embedding,
                    metadata=view
                )
                await self.index_view_embedding(view_embedding)
            
            logger.info(f"Successfully reindexed {len(views)} views")
            
        except Exception as e:
            logger.error(f"Error during reindex: {e}")
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = await self.client.indices.stats(index=self.index_name)
            doc_count = stats["indices"][self.index_name]["total"]["docs"]["count"]
            store_size = stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
            
            return {
                "document_count": doc_count,
                "store_size_bytes": store_size,
                "store_size_mb": round(store_size / 1024 / 1024, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}