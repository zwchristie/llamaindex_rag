"""MongoDB service for document management and storage."""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import structlog

from ..config.settings import settings

logger = structlog.get_logger(__name__)


class MongoDBService:
    """Service for managing documents in MongoDB."""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.documents_collection: Optional[Collection] = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to MongoDB instance."""
        try:
            # MongoDB connection string for Docker instance
            mongo_url = settings.mongodb.url
            database_name = settings.mongodb.database
            
            self.client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[database_name]
            self.documents_collection = self.db.documents
            
            # Create indexes for efficient querying
            self._create_indexes()
            
            logger.info(
                "Connected to MongoDB",
                database=database_name,
                url=mongo_url
            )
            
        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            # Don't raise exception - allow application to continue without MongoDB
            self.client = None
            self.db = None
            self.documents_collection = None
    
    def _create_indexes(self) -> None:
        """Create database indexes for efficient querying."""
        if self.documents_collection is None:
            return
        
        try:
            # Create indexes
            self.documents_collection.create_index("file_path", unique=True)
            self.documents_collection.create_index("document_type")
            self.documents_collection.create_index("catalog")
            self.documents_collection.create_index("schema_name")
            self.documents_collection.create_index("content_hash")
            self.documents_collection.create_index("last_modified")
            self.documents_collection.create_index([
                ("catalog", 1),
                ("schema_name", 1),
                ("document_type", 1)
            ])
            
            logger.info("Created MongoDB indexes")
            
        except Exception as e:
            logger.warning("Failed to create indexes", error=str(e))
    
    def is_connected(self) -> bool:
        """Check if MongoDB connection is active."""
        if self.client is None:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get document by file path."""
        if self.documents_collection is None:
            return None
        
        try:
            doc = self.documents_collection.find_one({"file_path": file_path})
            return doc
        except Exception as e:
            logger.error("Failed to get document by path", file_path=file_path, error=str(e))
            return None
    
    def upsert_document(
        self,
        file_path: str,
        content: str,
        document_type: str,
        catalog: str,
        schema_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert or update document in MongoDB."""
        if self.documents_collection is None:
            logger.warning("MongoDB not connected, skipping document upsert")
            return False
        
        try:
            content_hash = self.calculate_content_hash(content)
            
            document = {
                "file_path": file_path,
                "content": content,
                "content_hash": content_hash,
                "document_type": document_type,
                "catalog": catalog,
                "schema_name": schema_name,
                "metadata": metadata or {},
                "last_modified": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            # Upsert document (update if exists, insert if not)
            result = self.documents_collection.replace_one(
                {"file_path": file_path},
                document,
                upsert=True
            )
            
            if result.upserted_id:
                logger.info("Inserted new document", file_path=file_path, document_id=str(result.upserted_id))
            else:
                logger.info("Updated existing document", file_path=file_path)
            
            return True
            
        except Exception as e:
            logger.error("Failed to upsert document", file_path=file_path, error=str(e))
            return False
    
    def get_documents_by_catalog_schema(
        self,
        catalog: str,
        schema_name: str,
        document_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all documents for a specific catalog and schema."""
        if self.documents_collection is None:
            return []
        
        try:
            query = {
                "catalog": catalog,
                "schema_name": schema_name
            }
            
            if document_type:
                query["document_type"] = document_type
            
            documents = list(self.documents_collection.find(query))
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            return documents
            
        except Exception as e:
            logger.error(
                "Failed to get documents by catalog/schema",
                catalog=catalog,
                schema_name=schema_name,
                error=str(e)
            )
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from MongoDB."""
        if self.documents_collection is None:
            return []
        
        try:
            documents = list(self.documents_collection.find())
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            return documents
            
        except Exception as e:
            logger.error("Failed to get all documents", error=str(e))
            return []
    
    def check_document_needs_update(self, file_path: str, current_content: str) -> bool:
        """Check if document needs to be updated based on content hash."""
        existing_doc = self.get_document_by_path(file_path)
        
        if not existing_doc:
            return True  # Document doesn't exist, needs to be added
        
        current_hash = self.calculate_content_hash(current_content)
        existing_hash = existing_doc.get("content_hash", "")
        
        return current_hash != existing_hash
    
    def delete_document(self, file_path: str) -> bool:
        """Delete document by file path."""
        if self.documents_collection is None:
            return False
        
        try:
            result = self.documents_collection.delete_one({"file_path": file_path})
            
            if result.deleted_count > 0:
                logger.info("Deleted document", file_path=file_path)
                return True
            else:
                logger.warning("Document not found for deletion", file_path=file_path)
                return False
                
        except Exception as e:
            logger.error("Failed to delete document", file_path=file_path, error=str(e))
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.documents_collection is None:
            return {"error": "MongoDB not connected"}
        
        try:
            stats = self.db.command("collStats", "documents")
            
            # Get document counts by type
            pipeline = [
                {"$group": {
                    "_id": "$document_type",
                    "count": {"$sum": 1}
                }}
            ]
            
            type_counts = list(self.documents_collection.aggregate(pipeline))
            
            return {
                "total_documents": stats.get("count", 0),
                "storage_size": stats.get("storageSize", 0),
                "avg_document_size": stats.get("avgObjSize", 0),
                "document_counts_by_type": {
                    item["_id"]: item["count"] for item in type_counts
                }
            }
            
        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on MongoDB connection."""
        try:
            if self.client is None:
                return {
                    "status": "disconnected",
                    "message": "MongoDB client not initialized"
                }
            
            # Test connection with ping
            self.client.admin.command('ping')
            
            # Get basic stats
            stats = self.get_collection_stats()
            
            return {
                "status": "healthy",
                "connected": True,
                "database": self.db.name if self.db else None,
                "collection_stats": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e)
            }
    
    def close_connection(self) -> None:
        """Close MongoDB connection."""
        if self.client is not None:
            self.client.close()
            logger.info("Closed MongoDB connection")