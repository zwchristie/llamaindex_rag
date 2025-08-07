"""Document synchronization service for MongoDB and vector store."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import structlog

from .mongodb_service import MongoDBService
from .vector_service import LlamaIndexVectorService
from ..models.simple_models import (
    DocumentType, BusinessDomainMetadata, CoreViewMetadata, 
    SupportingViewMetadata, ReportMetadata
)
from ..models.meta_document import (
    DocumentSyncResult, SyncSummary, SchemaMetadata
)
from ..utils.content_processor import ContentProcessor

logger = structlog.get_logger(__name__)


class DocumentSyncService:
    """Service for synchronizing documents between filesystem, MongoDB, and vector store."""
    
    def __init__(
        self,
        mongodb_service: MongoDBService,
        vector_service: LlamaIndexVectorService,
        meta_documents_path: str = "meta_documents"
    ):
        self.mongodb_service = mongodb_service
        self.vector_service = vector_service
        self.meta_documents_path = Path(meta_documents_path)
        self.content_processor = ContentProcessor()
        
        # Ensure meta_documents directory exists
        self.meta_documents_path.mkdir(exist_ok=True)
    
    def sync_all_documents(self) -> SyncSummary:
        """Synchronize all documents from filesystem to MongoDB and vector store."""
        start_time = time.time()
        logger.info("Starting document synchronization", path=str(self.meta_documents_path))
        
        # Initialize summary
        summary = SyncSummary(
            total_files_processed=0,
            mongodb_operations={"created": 0, "updated": 0, "skipped": 0, "errors": 0},
            vector_store_operations={"added": 0, "updated": 0, "skipped": 0, "errors": 0},
            errors=[],
            duration_seconds=0.0
        )
        
        try:
            # Process all documents in meta_documents directory
            results = self._process_directory(self.meta_documents_path)
            
            for result in results:
                summary.total_files_processed += 1
                
                if result.success:
                    # Update MongoDB summary
                    if result.action in ["created", "updated", "skipped"]:
                        summary.mongodb_operations[result.action] += 1
                    
                    # Sync to vector store
                    vector_result = self._sync_to_vector_store(result.file_path)
                    if vector_result:
                        summary.vector_store_operations[vector_result] += 1
                    else:
                        summary.vector_store_operations["errors"] += 1
                        summary.errors.append(f"Vector store sync failed for {result.file_path}")
                else:
                    summary.mongodb_operations["errors"] += 1
                    if result.error:
                        summary.errors.append(f"{result.file_path}: {result.error}")
            
            summary.duration_seconds = time.time() - start_time
            
            logger.info(
                "Document synchronization completed",
                duration=summary.duration_seconds,
                total_files=summary.total_files_processed,
                mongodb_ops=summary.mongodb_operations,
                vector_ops=summary.vector_store_operations,
                errors_count=len(summary.errors)
            )
            
        except Exception as e:
            summary.duration_seconds = time.time() - start_time
            summary.errors.append(f"Synchronization failed: {str(e)}")
            logger.error("Document synchronization failed", error=str(e))
        
        return summary
    
    def _process_directory(self, directory_path: Path) -> List[DocumentSyncResult]:
        """Process all documents in a directory recursively."""
        results = []
        
        if not directory_path.exists():
            logger.warning("Directory does not exist", path=str(directory_path))
            return results
        
        # Process all JSON and text files
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.json', '.txt']:
                # Skip README files
                if file_path.name.lower().startswith('readme'):
                    continue
                
                result = self._process_single_file(file_path)
                results.append(result)
        
        return results
    
    def _process_single_file(self, file_path: Path) -> DocumentSyncResult:
        """Process a single document file."""
        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
            
            # Extract catalog and document type from path
            catalog, document_type, schema_name = self._extract_metadata_from_path(file_path)
            
            logger.info("Extracted metadata from path",
                       file_path=str(file_path),
                       catalog=catalog,
                       document_type=document_type.value if document_type else None,
                       schema_name=schema_name)
            
            if not catalog:
                return DocumentSyncResult(
                    file_path=str(file_path),
                    action="error",
                    success=False,
                    error="Could not extract catalog from file path"
                )
            
            # Check if document needs updating
            relative_path = str(file_path.relative_to(self.meta_documents_path))
            needs_update = self.mongodb_service.check_document_needs_update(relative_path, content)
            
            if not needs_update:
                return DocumentSyncResult(
                    file_path=relative_path,
                    action="skipped",
                    success=True,
                    message="Document is up to date"
                )
            
            # Parse and validate document content
            metadata = self._parse_document_content(content, document_type, catalog, schema_name)
            
            # Upsert to MongoDB
            # Handle both Pydantic models and plain dictionaries
            if metadata:
                if hasattr(metadata, 'model_dump'):
                    metadata_dict = metadata.model_dump()
                else:
                    metadata_dict = metadata
            else:
                metadata_dict = {}
                
            success = self.mongodb_service.upsert_document(
                file_path=relative_path,
                content=content,
                document_type=document_type.value,
                catalog=catalog,
                schema_name=schema_name,
                metadata=metadata_dict
            )
            
            if success:
                action = "updated" if not needs_update else "created"
                return DocumentSyncResult(
                    file_path=relative_path,
                    action=action,
                    success=True,
                    message=f"Document {action} successfully"
                )
            else:
                return DocumentSyncResult(
                    file_path=relative_path,
                    action="error",
                    success=False,
                    error="Failed to upsert document to MongoDB"
                )
                
        except Exception as e:
            logger.error("Failed to process file", file_path=str(file_path), error=str(e))
            return DocumentSyncResult(
                file_path=str(file_path),
                action="error",
                success=False,
                error=str(e)
            )
    
    def _extract_metadata_from_path(self, file_path: Path) -> Tuple[Optional[str], Optional[DocumentType], Optional[str]]:
        """Extract catalog, document type, and schema name from file path."""
        try:
            # Get relative path from meta_documents directory
            relative_path = file_path.relative_to(self.meta_documents_path)
            parts = relative_path.parts
            
            logger.info("Parsing file path for metadata extraction",
                       file_path=str(file_path),
                       relative_path=str(relative_path),
                       parts=list(parts),
                       parts_count=len(parts))
            
            if len(parts) < 2:
                logger.warning("Path has insufficient parts", parts=list(parts))
                return None, None, None
            
            catalog = parts[0]
            
            # Determine document type from directory structure (business domain-first architecture)
            if len(parts) >= 3 and parts[1] == "schema":
                if len(parts) >= 4 and parts[2] == "ddl":
                    # DDL files: /catalog/schema/ddl/view_name.sql
                    if file_path.suffix.lower() == '.sql':
                        document_type = DocumentType.DDL
                        schema_name = file_path.stem  # view name
                    else:
                        return None, None, None
                else:
                    # Legacy schema files: /catalog/schema/main_schema_metadata.json
                    document_type = DocumentType.SCHEMA
                    schema_name = file_path.stem.replace('_metadata', '')
            
            # New business domain-first structure
            elif len(parts) >= 4 and parts[1] == "processed_views":
                if parts[2] == "core_views":
                    # Core views: /catalog/processed_views/core_views/view_name.json
                    if file_path.suffix.lower() == '.json':
                        document_type = DocumentType.CORE_VIEW
                        schema_name = file_path.stem
                    else:
                        return None, None, None
                elif parts[2] == "supporting_views":
                    # Supporting views: /catalog/processed_views/supporting_views/view_name.json
                    if file_path.suffix.lower() == '.json':
                        document_type = DocumentType.SUPPORTING_VIEW
                        schema_name = file_path.stem
                    else:
                        return None, None, None
                else:
                    return None, None, None
            
            elif len(parts) >= 4 and parts[1] == "processed_reports":
                if parts[2] == "processed_reports":
                    # Processed reports: /catalog/processed_reports/processed_reports/report_name.json
                    if file_path.suffix.lower() == '.json':
                        document_type = DocumentType.REPORT
                        schema_name = file_path.stem
                    else:
                        return None, None, None
                else:
                    return None, None, None
            
            elif len(parts) >= 4 and parts[1] == "processed_domains":
                if parts[2] == "business_domains":
                    # Business domains: /catalog/processed_domains/business_domains/domain_name.json
                    if file_path.suffix.lower() == '.json':
                        document_type = DocumentType.BUSINESS_DOMAIN
                        schema_name = file_path.stem
                    else:
                        return None, None, None
                else:
                    return None, None, None
            
            # Legacy paths
            elif len(parts) >= 3 and parts[1] == "columns":
                # Legacy column details: /catalog/columns/table_name.json
                if file_path.suffix.lower() == '.json':
                    document_type = DocumentType.COLUMN_DETAILS
                    schema_name = file_path.stem
                else:
                    return None, None, None
            elif len(parts) >= 3 and parts[1] == "lookups":
                # Lookup metadata: /catalog/lookups/lookup_name.json
                if file_path.suffix.lower() == '.json':
                    document_type = DocumentType.LOOKUP_METADATA
                    schema_name = "lookups"
                else:
                    return None, None, None
            elif len(parts) >= 3 and parts[1] == "reports":
                # Legacy reports: /catalog/reports/report_name.txt/.md
                document_type = DocumentType.REPORT
                schema_name = "reports"
            else:
                # Legacy fallback - try to infer from file content or name
                if file_path.suffix.lower() == '.json':
                    document_type = DocumentType.SCHEMA
                    schema_name = file_path.stem.replace('_metadata', '')
                else:
                    document_type = DocumentType.REPORT
                    schema_name = "reports"
            
            return catalog, document_type, schema_name
            
        except Exception as e:
            logger.error("Failed to extract metadata from path", file_path=str(file_path), error=str(e))
            return None, None, None
    
    def _parse_document_content(
        self,
        content: str,
        document_type: DocumentType,
        catalog: str,
        schema_name: str
    ) -> Optional[Any]:
        """Parse document content based on type."""
        try:
            # Legacy types
            if document_type == DocumentType.SCHEMA:
                # Parse JSON schema document (legacy)
                data = json.loads(content)
                return SchemaMetadata(**data)
            
            elif document_type == DocumentType.REPORT:
                # Parse text report document
                return self._parse_report_content(content, catalog)
            
            # New business domain-first types
            elif document_type == DocumentType.BUSINESS_DOMAIN:
                # Business domain JSON files
                data = json.loads(content)
                return data
            
            elif document_type == DocumentType.CORE_VIEW:
                # Core view JSON files
                data = json.loads(content)
                return data
            
            elif document_type == DocumentType.SUPPORTING_VIEW:
                # Supporting view JSON files
                data = json.loads(content)
                return data
            
            # Legacy and supporting types
            elif document_type == DocumentType.DDL:
                # DDL files are plain SQL - return as simple metadata
                return {
                    "view_name": schema_name,
                    "ddl_content": content,
                    "file_type": "sql"
                }
            
            elif document_type == DocumentType.COLUMN_DETAILS:
                # Legacy column details JSON files
                data = json.loads(content)
                return data
            
            elif document_type == DocumentType.LOOKUP_METADATA:
                # Parse JSON lookup metadata document
                data = json.loads(content)
                return data
            
        except Exception as e:
            logger.error(
                "Failed to parse document content",
                document_type=document_type.value,
                error=str(e)
            )
            return None
    
    def _parse_report_content(self, content: str, catalog: str) -> Optional[ReportMetadata]:
        """Parse report text content to extract structured information."""
        try:
            # Extract report name from first line or content
            lines = content.strip().split('\n')
            report_name = lines[0].strip() if lines else "Unknown Report"
            
            # Extract description
            description = ""
            sql_query = ""
            data_returned = ""
            use_cases = ""
            
            current_section = None
            section_content = []
            
            for line in lines[1:]:  # Skip first line (report name)
                line = line.strip()
                
                # Detect section headers
                if line.lower().startswith('description:'):
                    current_section = 'description'
                    section_content = []
                    continue
                elif line.lower().startswith('data returned:'):
                    if current_section == 'description':
                        description = '\n'.join(section_content).strip()
                    current_section = 'data_returned'
                    section_content = []
                    continue
                elif line.lower().startswith('sql query:') or line.lower().startswith('sql:'):
                    if current_section == 'data_returned':
                        data_returned = '\n'.join(section_content).strip()
                    current_section = 'sql_query'
                    section_content = []
                    continue
                elif line.lower().startswith('use cases:'):
                    if current_section == 'sql_query':
                        sql_query = '\n'.join(section_content).strip()
                    current_section = 'use_cases'
                    section_content = []
                    continue
                
                # Add content to current section
                if current_section and line:
                    section_content.append(line)
            
            # Handle last section
            if current_section == 'description':
                description = '\n'.join(section_content).strip()
            elif current_section == 'data_returned':
                data_returned = '\n'.join(section_content).strip()
            elif current_section == 'sql_query':
                sql_query = '\n'.join(section_content).strip()
            elif current_section == 'use_cases':
                use_cases = '\n'.join(section_content).strip()
            
            # Clean up SQL query (remove comments and extra whitespace)
            if sql_query:
                sql_query = self._clean_sql_query(sql_query)
            
            return ReportMetadata(
                catalog=catalog,
                report_name=report_name,
                description=description or "No description available",
                sql_query=sql_query or "-- No SQL query found",
                data_returned=data_returned or "No data description available",
                use_cases=use_cases
            )
            
        except Exception as e:
            logger.error("Failed to parse report content", error=str(e))
            return None
    
    def _check_vector_store_needs_update(self, document_id: str, mongo_doc: dict) -> bool:
        """Check if vector store needs updating for this document."""
        try:
            # Get document info from vector store
            doc_info = self.vector_service.get_document_info(document_id)
            
            # If document doesn't exist in vector store, it needs to be added
            if doc_info.get("status") == "not_found" or doc_info.get("num_chunks", 0) == 0:
                logger.info("Document not found in vector store, needs indexing", document_id=document_id)
                return True
            
            # Compare content hashes to determine if update is needed
            mongo_content_hash = mongo_doc.get("content_hash")
            vector_metadata = doc_info.get("metadata", {})
            vector_content_hash = vector_metadata.get("content_hash")
            
            if not mongo_content_hash:
                logger.info("No content hash available in MongoDB, assuming update needed", document_id=document_id)
                return True
            
            if mongo_content_hash != vector_content_hash:
                logger.info("Content hash changed, update needed", 
                           document_id=document_id,
                           mongo_hash=mongo_content_hash,
                           vector_hash=vector_content_hash)
                return True
            
            logger.info("Vector store is up to date (content hash matches)", 
                       document_id=document_id, 
                       content_hash=mongo_content_hash)
            return False
            
        except Exception as e:
            logger.warning("Error checking vector store status, assuming update needed", 
                         document_id=document_id, error=str(e))
            return True
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and format SQL query."""
        # Remove line comments
        lines = []
        for line in sql_query.split('\n'):
            # Remove comments but keep the line
            line = re.sub(r'--.*$', '', line).strip()
            if line:
                lines.append(line)
        
        # Join lines and normalize whitespace
        cleaned = ' '.join(lines)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _sync_to_vector_store(self, file_path: str) -> Optional[str]:
        """Sync document to vector store if needed."""
        try:
            # Get document from MongoDB
            mongo_doc = self.mongodb_service.get_document_by_path(file_path)
            if not mongo_doc:
                logger.warning("Document not found in MongoDB", file_path=file_path)
                return None
            
            # file_path is already a relative path string from _process_single_file
            # Convert to Path object for document ID extraction
            relative_path_obj = Path(file_path)
            document_id = relative_path_obj.stem  # e.g., "main_schema_metadata" or "adf_report"
            
            # file_path is already the relative path string we need
            relative_path = file_path
            
            # Check if document needs updating in vector store
            # Skip version checking if vector store is empty (first run)
            try:
                vector_needs_update = self._check_vector_store_needs_update(document_id, mongo_doc)
                
                if not vector_needs_update:
                    logger.info("Vector store is up to date", document_id=document_id, file_path=relative_path)
                    return "skipped"
            except Exception as e:
                logger.info("Version checking failed, assuming update needed", 
                           document_id=document_id, 
                           error=str(e))
                vector_needs_update = True
            
            # Use original content - vector service will handle semantic chunking
            content = mongo_doc["content"]
            
            # For simplicity during initial indexing, just try to add without deleting
            # The vector store should handle overwrites automatically
            logger.info("Adding document to vector store", document_id=document_id)
            
            # Add/update document in vector store with content hash for versioning
            vector_metadata = {
                **mongo_doc.get("metadata", {}),
                "content_hash": mongo_doc.get("content_hash"),  # Include content hash for version tracking
                "updated_at": mongo_doc.get("updated_at"),
                "file_path": relative_path
            }
            
            success = self.vector_service.add_document(
                document_id=document_id,
                content=content,
                metadata=vector_metadata,
                document_type=mongo_doc["document_type"]
            )
            
            if success:
                return "added"  # We're treating all as "added" since we don't track vector store versions
            else:
                return None
                
        except Exception as e:
            logger.error("Failed to sync to vector store", file_path=file_path, error=str(e))
            return None
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        try:
            # Get MongoDB stats
            mongodb_stats = self.mongodb_service.get_collection_stats()
            
            # Get vector store stats
            vector_stats = self.vector_service.get_index_stats()
            
            # Check file system
            file_count = 0
            if self.meta_documents_path.exists():
                file_count = len([
                    f for f in self.meta_documents_path.rglob("*")
                    if f.is_file() and f.suffix.lower() in ['.json', '.txt'] 
                    and not f.name.lower().startswith('readme')
                ])
            
            return {
                "filesystem": {
                    "meta_documents_path": str(self.meta_documents_path),
                    "total_files": file_count
                },
                "mongodb": mongodb_stats,
                "vector_store": vector_stats,
                "services_status": {
                    "mongodb_connected": self.mongodb_service.is_connected(),
                    "vector_store_healthy": self.vector_service.health_check()
                }
            }
            
        except Exception as e:
            logger.error("Failed to get sync status", error=str(e))
            return {"error": str(e)}