"""Document management service with versioning and deduplication."""

import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from ..models.document import (
    Document, DocumentVersion, DocumentType, DocumentStatus,
    DocumentCreate, DocumentUpdate, DocumentResponse
)
from ..utils.content_processor import ContentProcessor


class DocumentService:
    """Service for managing documents with versioning and deduplication."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.content_processor = ContentProcessor()
    
    def _calculate_content_hash(self, content: str, metadata: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash of content and metadata for deduplication."""
        combined_content = {
            "content": content.strip(),
            "metadata": metadata
        }
        content_str = json.dumps(combined_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _create_document_version(
        self, 
        document_id: int, 
        version_number: int,
        content: str, 
        metadata: Dict[str, Any],
        content_hash: str,
        change_summary: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> DocumentVersion:
        """Create a new document version."""
        version = DocumentVersion(
            document_id=document_id,
            version_number=version_number,
            content_hash=content_hash,
            content=content,
            metadata=metadata,
            change_summary=change_summary,
            created_by=created_by
        )
        self.db.add(version)
        return version
    
    def create_document(
        self, 
        document_data: DocumentCreate, 
        file_path: str,
        uploaded_by: Optional[str] = None
    ) -> Tuple[DocumentResponse, bool]:
        """
        Create a new document or update existing one if content is different.
        
        Returns:
            Tuple of (DocumentResponse, is_new_document)
        """
        content_hash = self._calculate_content_hash(
            document_data.content, 
            document_data.metadata
        )
        
        # Check for existing document with same title and type
        existing_doc = self.db.query(Document).filter(
            and_(
                Document.title == document_data.title,
                Document.document_type == document_data.document_type,
                Document.is_active == True
            )
        ).first()
        
        if existing_doc:
            # Check if content has changed
            if existing_doc.content_hash == content_hash:
                # Content is identical, return existing document
                return DocumentResponse.from_orm(existing_doc), False
            else:
                # Content has changed, create new version
                return self._update_document_with_version(
                    existing_doc, 
                    document_data.content,
                    document_data.metadata,
                    content_hash,
                    file_path,
                    uploaded_by
                ), False
        else:
            # Create new document
            document = Document(
                title=document_data.title,
                document_type=document_data.document_type,
                status=DocumentStatus.UPLOADED,
                file_path=file_path,
                content_hash=content_hash,
                metadata=document_data.metadata,
                content=document_data.content,
                version=1,
                uploaded_by=uploaded_by
            )
            
            self.db.add(document)
            self.db.flush()  # Get the ID
            
            # Create initial version
            self._create_document_version(
                document_id=document.id,
                version_number=1,
                content=document_data.content,
                metadata=document_data.metadata,
                content_hash=content_hash,
                change_summary="Initial version",
                created_by=uploaded_by
            )
            
            self.db.commit()
            return DocumentResponse.from_orm(document), True
    
    def _update_document_with_version(
        self,
        existing_doc: Document,
        new_content: str,
        new_metadata: Dict[str, Any],
        content_hash: str,
        file_path: str,
        uploaded_by: Optional[str] = None
    ) -> DocumentResponse:
        """Update existing document with new version."""
        
        # Create new version
        new_version_number = existing_doc.version + 1
        self._create_document_version(
            document_id=existing_doc.id,
            version_number=new_version_number,
            content=new_content,
            metadata=new_metadata,
            content_hash=content_hash,
            change_summary=f"Updated content (version {new_version_number})",
            created_by=uploaded_by
        )
        
        # Update main document record
        existing_doc.content = new_content
        existing_doc.metadata = new_metadata
        existing_doc.content_hash = content_hash
        existing_doc.file_path = file_path
        existing_doc.version = new_version_number
        existing_doc.status = DocumentStatus.UPLOADED
        existing_doc.updated_at = datetime.utcnow()
        existing_doc.uploaded_by = uploaded_by
        
        self.db.commit()
        return DocumentResponse.from_orm(existing_doc)
    
    def get_document(self, document_id: int) -> Optional[DocumentResponse]:
        """Get document by ID."""
        document = self.db.query(Document).filter(
            and_(Document.id == document_id, Document.is_active == True)
        ).first()
        
        if document:
            return DocumentResponse.from_orm(document)
        return None
    
    def get_document_by_title_and_type(
        self, 
        title: str, 
        document_type: DocumentType
    ) -> Optional[DocumentResponse]:
        """Get document by title and type."""
        document = self.db.query(Document).filter(
            and_(
                Document.title == title,
                Document.document_type == document_type,
                Document.is_active == True
            )
        ).first()
        
        if document:
            return DocumentResponse.from_orm(document)
        return None
    
    def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentResponse]:
        """List documents with optional filtering."""
        query = self.db.query(Document).filter(Document.is_active == True)
        
        if document_type:
            query = query.filter(Document.document_type == document_type)
        
        if status:
            query = query.filter(Document.status == status)
        
        documents = query.order_by(desc(Document.updated_at)).offset(offset).limit(limit).all()
        return [DocumentResponse.from_orm(doc) for doc in documents]
    
    def update_document(
        self, 
        document_id: int, 
        update_data: DocumentUpdate,
        updated_by: Optional[str] = None
    ) -> Optional[DocumentResponse]:
        """Update document with change tracking."""
        document = self.db.query(Document).filter(
            and_(Document.id == document_id, Document.is_active == True)
        ).first()
        
        if not document:
            return None
        
        # Check if content or metadata changed
        content_changed = False
        if update_data.content and update_data.content != document.content:
            content_changed = True
        
        metadata_changed = False
        if update_data.metadata and update_data.metadata != document.metadata:
            metadata_changed = True
        
        if content_changed or metadata_changed:
            # Create new version for significant changes
            new_content = update_data.content or document.content
            new_metadata = update_data.metadata or document.metadata
            content_hash = self._calculate_content_hash(new_content, new_metadata)
            
            new_version_number = document.version + 1
            self._create_document_version(
                document_id=document.id,
                version_number=new_version_number,
                content=new_content,
                metadata=new_metadata,
                content_hash=content_hash,
                change_summary=update_data.change_summary or "Content updated",
                created_by=updated_by
            )
            
            # Update document
            document.content = new_content
            document.metadata = new_metadata
            document.content_hash = content_hash
            document.version = new_version_number
        
        # Update other fields
        if update_data.title:
            document.title = update_data.title
        
        document.updated_at = datetime.utcnow()
        
        self.db.commit()
        return DocumentResponse.from_orm(document)
    
    def delete_document(self, document_id: int, soft_delete: bool = True) -> bool:
        """Delete document (soft delete by default)."""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            return False
        
        if soft_delete:
            document.is_active = False
            document.status = DocumentStatus.SUPERSEDED
            document.updated_at = datetime.utcnow()
        else:
            self.db.delete(document)
        
        self.db.commit()
        return True
    
    def get_document_versions(self, document_id: int) -> List[DocumentVersion]:
        """Get all versions of a document."""
        return self.db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).order_by(desc(DocumentVersion.version_number)).all()
    
    def get_document_version(
        self, 
        document_id: int, 
        version_number: int
    ) -> Optional[DocumentVersion]:
        """Get specific version of a document."""
        return self.db.query(DocumentVersion).filter(
            and_(
                DocumentVersion.document_id == document_id,
                DocumentVersion.version_number == version_number
            )
        ).first()
    
    def find_duplicates(self) -> Dict[str, List[DocumentResponse]]:
        """Find documents with identical content hashes."""
        documents = self.db.query(Document).filter(Document.is_active == True).all()
        
        hash_groups = {}
        for doc in documents:
            if doc.content_hash not in hash_groups:
                hash_groups[doc.content_hash] = []
            hash_groups[doc.content_hash].append(DocumentResponse.from_orm(doc))
        
        # Return only groups with more than one document
        return {h: docs for h, docs in hash_groups.items() if len(docs) > 1}
    
    def update_document_status(
        self, 
        document_id: int, 
        status: DocumentStatus
    ) -> bool:
        """Update document processing status."""
        document = self.db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            return False
        
        document.status = status
        if status == DocumentStatus.INDEXED:
            document.indexed_at = datetime.utcnow()
        
        self.db.commit()
        return True