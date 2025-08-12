"""Application startup logic for document synchronization."""

import asyncio
from pathlib import Path
from typing import Optional

import structlog

from ..services.mongodb_service import MongoDBService
from ..services.vector_service import LlamaIndexVectorService
from ..services.llm_provider_factory import llm_factory
from ..config.settings import settings

logger = structlog.get_logger(__name__)


class ApplicationStartup:
    """Handles application startup tasks including document synchronization."""
    
    def __init__(self):
        self.mongodb_service: Optional[MongoDBService] = None
        self.vector_service: Optional[LlamaIndexVectorService] = None
        self.sync_service: Optional = None  # Legacy - no longer used
    
    async def initialize_services(self) -> bool:
        """Initialize all required services."""
        services_initialized = 0
        total_services = 3  # MongoDB, Vector Store, LLM Provider
        
        logger.info("Initializing application services")
        
        # Initialize MongoDB service (non-critical)
        try:
            logger.info("Connecting to MongoDB")
            self.mongodb_service = MongoDBService()
            
            if not self.mongodb_service.is_connected():
                logger.warning("MongoDB connection failed - continuing without MongoDB")
            else:
                logger.info("MongoDB connected successfully")
                services_initialized += 1
        except Exception as e:
            logger.error("Failed to initialize MongoDB service", error=str(e))
            self.mongodb_service = None
        
        # Initialize vector service (critical)
        try:
            logger.info("Initializing vector store service")
            self.vector_service = LlamaIndexVectorService()
            
            if not self.vector_service.health_check():
                logger.error("Vector store service health check failed")
                logger.warning("Application will continue with limited functionality")
            else:
                logger.info("Vector store service initialized successfully")
                services_initialized += 1
        except Exception as e:
            logger.error("Failed to initialize vector store service", error=str(e))
            logger.warning("Vector store functionality will not be available")
            # Don't return False - let the app continue with limited functionality
            self.vector_service = None
        
        # Check LLM provider configuration (critical for SQL generation)
        try:
            logger.info(f"Checking LLM provider configuration ({settings.llm_provider.provider})")
            llm_configured = llm_factory.is_configured()
            if not llm_configured:
                logger.warning(f"LLM provider ({settings.llm_provider.provider}) configuration check failed")
                logger.warning("SQL generation functionality may be limited")
            else:
                provider_info = llm_factory.get_provider_info()
                logger.info(f"LLM Provider: {provider_info['provider']} - Configuration: OK", provider_info=provider_info)
                services_initialized += 1
        except Exception as e:
            logger.error("Failed to initialize LLM provider", error=str(e))
            logger.warning("LLM functionality may be limited")
        
        # Document sync service is no longer needed - we sync directly from MongoDB to Vector Store
        self.sync_service = None
        logger.info("Using direct MongoDB to Vector Store synchronization")
        
        logger.info(f"Service initialization completed: {services_initialized}/{total_services} services initialized")
        
        # Application can continue with at least partial functionality
        return True
    
    async def sync_documents(self) -> bool:
        """Perform document synchronization on startup - MongoDB to Vector Store."""
        if not self.mongodb_service or not self.vector_service:
            logger.warning("MongoDB or Vector service not available - skipping document embedding")
            return True  # Not an error, just no services available
        
        try:
            logger.info("Starting MongoDB to Vector Store synchronization")
            
            # Get all documents from MongoDB
            mongo_docs = self.mongodb_service.get_all_documents()
            
            if not mongo_docs:
                logger.info("No documents found in MongoDB to embed")
                return True
            
            logger.info(f"Found {len(mongo_docs)} documents in MongoDB to process")
            
            # Embed documents that need it
            embedded_count = 0
            error_count = 0
            
            for mongo_doc in mongo_docs:
                try:
                    document_id = mongo_doc.get("schema_name", mongo_doc.get("file_path", "unknown"))
                    
                    # Check if document needs embedding (not already in vector store)
                    doc_info = self.vector_service.get_document_info(document_id)
                    
                    # Only embed if document is not found or content has changed
                    mongo_content_hash = mongo_doc.get("content_hash")
                    vector_content_hash = doc_info.get("metadata", {}).get("content_hash")
                    
                    if (doc_info.get("status") == "not_found" or 
                        mongo_content_hash != vector_content_hash):
                        
                        logger.info(f"Embedding document: {document_id}")
                        
                        # Prepare metadata for vector store
                        vector_metadata = {
                            **mongo_doc.get("metadata", {}),
                            "content_hash": mongo_content_hash,
                            "updated_at": mongo_doc.get("updated_at"),
                            "file_path": mongo_doc.get("file_path"),
                            "document_type": mongo_doc.get("document_type"),
                            "catalog": mongo_doc.get("catalog"),
                            "schema_name": mongo_doc.get("schema_name")
                        }
                        
                        # Add document to vector store
                        success = self.vector_service.add_document(
                            document_id=document_id,
                            content=mongo_doc["content"],
                            metadata=vector_metadata,
                            document_type=mongo_doc["document_type"]
                        )
                        
                        if success:
                            embedded_count += 1
                        else:
                            error_count += 1
                            logger.error(f"Failed to embed document: {document_id}")
                    else:
                        logger.debug(f"Document up to date: {document_id}")
                        
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing document", error=str(e))
            
            logger.info(
                "MongoDB to Vector Store synchronization completed",
                total_documents=len(mongo_docs),
                embedded=embedded_count,
                errors=error_count
            )
            
            return error_count == 0
            
        except Exception as e:
            logger.error("Document synchronization failed", error=str(e))
            return False
    
    async def perform_startup_tasks(self) -> bool:
        """Perform all startup tasks."""
        try:
            logger.info("Starting application startup tasks")
            
            # Initialize services (always succeeds with partial functionality)
            services_ok = await self.initialize_services()
            logger.info(f"Service initialization result: {services_ok}")
            
            # Sync documents (optional)
            sync_ok = await self.sync_documents()
            if not sync_ok:
                logger.warning("Document synchronization had errors - continuing anyway")
            
            # Log service availability summary
            available_services = []
            if self.mongodb_service and self.mongodb_service.is_connected():
                available_services.append("MongoDB")
            if self.vector_service:
                available_services.append("Vector Store")
            
            # Get status from individual services
            if self.mongodb_service:
                try:
                    mongo_status = self.mongodb_service.health_check()
                    logger.info("MongoDB status", status=mongo_status)
                except Exception as e:
                    logger.warning("Failed to get MongoDB status", error=str(e))
            
            if self.vector_service:
                try:
                    vector_status = self.vector_service.get_index_stats()
                    logger.info("Vector Store status", status=vector_status)
                except Exception as e:
                    logger.warning("Failed to get Vector Store status", error=str(e))
            
            logger.info("Application startup completed", available_services=available_services)
            return True
            
        except Exception as e:
            logger.error("Startup tasks failed", error=str(e))
            # Even if startup tasks fail, don't prevent the application from starting
            logger.warning("Continuing with limited functionality")
            return True
    
    def get_services(self) -> tuple:
        """Get initialized services."""
        return self.mongodb_service, self.vector_service, None


# Global startup instance
startup = ApplicationStartup()


async def run_startup_tasks() -> bool:
    """Run application startup tasks."""
    return await startup.perform_startup_tasks()


def get_initialized_services() -> tuple:
    """Get initialized services after startup."""
    return startup.get_services()