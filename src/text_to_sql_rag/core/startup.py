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
        try:
            logger.info("Initializing application services")
            
            # Initialize MongoDB service
            logger.info("Connecting to MongoDB")
            self.mongodb_service = MongoDBService()
            
            if not self.mongodb_service.is_connected():
                logger.warning("MongoDB connection failed - continuing without MongoDB")
            else:
                logger.info("MongoDB connected successfully")
            
            # Initialize vector service
            logger.info("Initializing vector store service")
            self.vector_service = LlamaIndexVectorService()
            
            if not self.vector_service.health_check():
                logger.error("Vector store service health check failed")
                return False
            else:
                logger.info("Vector store service initialized successfully")
            
            # Check LLM provider health
            logger.info(f"Checking LLM provider health ({settings.llm_provider.provider})")
            llm_healthy = llm_factory.health_check()
            if not llm_healthy:
                logger.warning(f"LLM provider ({settings.llm_provider.provider}) health check failed")
            else:
                provider_info = llm_factory.get_provider_info()
                logger.info(f"LLM Provider: {provider_info['provider']} - Health: OK", provider_info=provider_info)
            
            # Initialize document sync service
            meta_docs_path = settings.app.meta_documents_path
            self.sync_service = DocumentSyncService(
                mongodb_service=self.mongodb_service,
                vector_service=self.vector_service,
                meta_documents_path=meta_docs_path
            )
            
            logger.info("All services initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize services", error=str(e))
            return False
    
    async def sync_documents(self) -> bool:
        """Perform document synchronization on startup."""
        if not self.sync_service:
            logger.error("Document sync service not initialized")
            return False
        
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
            
            # Initialize services
            services_ok = await self.initialize_services()
            if not services_ok:
                logger.error("Service initialization failed")
                return False
            
            # Sync documents
            sync_ok = await self.sync_documents()
            if not sync_ok:
                logger.warning("Document synchronization had errors - continuing anyway")
            
            # Get and log sync status
            if self.sync_service:
                status = self.sync_service.get_sync_status()
                logger.info("Startup sync status", status=status)
            
            logger.info("Application startup tasks completed successfully")
            return True
            
        except Exception as e:
            logger.error("Startup tasks failed", error=str(e))
            return False
    
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