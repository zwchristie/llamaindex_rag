"""Application startup logic for document synchronization."""

import asyncio
from pathlib import Path
from typing import Optional

import structlog

from ..services.mongodb_service import MongoDBService
from ..services.vector_service import LlamaIndexVectorService
from ..services.document_sync_service import DocumentSyncService
from ..services.llm_provider_factory import llm_factory
from ..config.settings import settings

logger = structlog.get_logger(__name__)


class ApplicationStartup:
    """Handles application startup tasks including document synchronization."""
    
    def __init__(self):
        self.mongodb_service: Optional[MongoDBService] = None
        self.vector_service: Optional[LlamaIndexVectorService] = None
        self.sync_service: Optional[DocumentSyncService] = None
    
    async def initialize_services(self) -> bool:
        """Initialize all required services."""
        services_initialized = 0
        total_services = 4  # MongoDB, Vector Store, LLM Provider, Sync Service
        
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
        
        # Check LLM provider health (critical for SQL generation)
        try:
            logger.info(f"Checking LLM provider health ({settings.llm_provider.provider})")
            llm_healthy = llm_factory.health_check()
            if not llm_healthy:
                logger.warning(f"LLM provider ({settings.llm_provider.provider}) health check failed")
                logger.warning("SQL generation functionality may be limited")
            else:
                provider_info = llm_factory.get_provider_info()
                logger.info(f"LLM Provider: {provider_info['provider']} - Health: OK", provider_info=provider_info)
                services_initialized += 1
        except Exception as e:
            logger.error("Failed to initialize LLM provider", error=str(e))
            logger.warning("LLM functionality may be limited")
        
        # Initialize document sync service (depends on MongoDB and Vector store)
        try:
            if self.mongodb_service or self.vector_service:
                meta_docs_path = settings.app.meta_documents_path
                self.sync_service = DocumentSyncService(
                    mongodb_service=self.mongodb_service,
                    vector_service=self.vector_service,
                    meta_documents_path=meta_docs_path
                )
                logger.info("Document sync service initialized successfully")
                services_initialized += 1
            else:
                logger.warning("Skipping document sync service - no storage backends available")
        except Exception as e:
            logger.error("Failed to initialize document sync service", error=str(e))
            self.sync_service = None
        
        logger.info(f"Service initialization completed: {services_initialized}/{total_services} services initialized")
        
        # Application can continue with at least partial functionality
        return True
    
    async def sync_documents(self) -> bool:
        """Perform document synchronization on startup."""
        if self.sync_service is None:
            logger.warning("Document sync service not available - skipping synchronization")
            return True  # Not an error, just no sync service available
        
        try:
            logger.info("Starting document synchronization")
            
            # Check if meta_documents directory exists and has files
            meta_docs_path = Path(self.sync_service.meta_documents_path)
            if not meta_docs_path.exists():
                logger.warning(
                    "Meta documents directory does not exist", 
                    path=str(meta_docs_path)
                )
                return True  # Not an error, just no documents to sync
            
            # Count files to sync
            files_to_sync = [
                f for f in meta_docs_path.rglob("*")
                if f.is_file() and f.suffix.lower() in ['.json', '.txt']
                and not f.name.lower().startswith('readme')
            ]
            
            if not files_to_sync:
                logger.info("No metadata documents found to synchronize")
                return True
            
            logger.info(f"Found {len(files_to_sync)} metadata documents to synchronize")
            
            # Perform synchronization
            sync_summary = self.sync_service.sync_all_documents()
            
            # Log synchronization results
            logger.info(
                "Document synchronization completed",
                total_files=sync_summary.total_files_processed,
                duration=sync_summary.duration_seconds,
                mongodb_ops=sync_summary.mongodb_operations,
                vector_ops=sync_summary.vector_store_operations,
                errors_count=len(sync_summary.errors)
            )
            
            # Log errors if any
            if sync_summary.errors:
                for error in sync_summary.errors:
                    logger.error("Sync error", error=error)
                return False
            
            return True
            
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
            
            # Get and log sync status if service is available
            if self.sync_service:
                try:
                    status = self.sync_service.get_sync_status()
                    logger.info("Startup sync status", status=status)
                except Exception as e:
                    logger.warning("Failed to get sync status", error=str(e))
            
            # Log service availability summary
            available_services = []
            if self.mongodb_service and self.mongodb_service.is_connected():
                available_services.append("MongoDB")
            if self.vector_service:
                available_services.append("Vector Store")
            if self.sync_service:
                available_services.append("Document Sync")
            
            logger.info("Application startup completed", available_services=available_services)
            return True
            
        except Exception as e:
            logger.error("Startup tasks failed", error=str(e))
            # Even if startup tasks fail, don't prevent the application from starting
            logger.warning("Continuing with limited functionality")
            return True
    
    def get_services(self) -> tuple:
        """Get initialized services."""
        return self.mongodb_service, self.vector_service, self.sync_service


# Global startup instance
startup = ApplicationStartup()


async def run_startup_tasks() -> bool:
    """Run application startup tasks."""
    return await startup.perform_startup_tasks()


def get_initialized_services() -> tuple:
    """Get initialized services after startup."""
    return startup.get_services()