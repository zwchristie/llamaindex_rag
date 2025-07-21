"""Document models for the text-to-SQL RAG system."""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, Index
from sqlalchemy.sql import func

from .simple_models import DocumentType
from .database import Base


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class Document(Base):
    """Database model for documents."""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    document_type = Column(String(50), nullable=False, index=True)
    status = Column(String(50), default=DocumentStatus.UPLOADED, index=True)
    file_path = Column(String(500), nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    metadata = Column(JSON, default={})
    content = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    
    # User tracking
    uploaded_by = Column(String(255), nullable=True)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_document_type_status', 'document_type', 'status'),
        Index('idx_content_hash_active', 'content_hash', 'is_active'),
        Index('idx_title_type', 'title', 'document_type'),
    )


class DocumentVersion(Base):
    """Track document versions for change management."""
    
    __tablename__ = "document_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSON, default={})
    change_summary = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255), nullable=True)
    
    __table_args__ = (
        Index('idx_document_version', 'document_id', 'version_number'),
    )


# Pydantic models for API

class DocumentMetadata(BaseModel):
    """Base metadata for all document types."""
    title: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ReportDocumentMetadata(DocumentMetadata):
    """Metadata specific to report documents."""
    sql_query: str
    expected_output_description: str
    complexity_level: Optional[str] = None
    use_cases: List[str] = Field(default_factory=list)
    related_tables: List[str] = Field(default_factory=list)


class SchemaDocumentMetadata(DocumentMetadata):
    """Metadata specific to schema documents."""
    table_name: str
    columns: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    indexes: List[str] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentCreate(BaseModel):
    """Model for creating a new document."""
    title: str
    document_type: DocumentType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_name: Optional[str] = None


class DocumentUpdate(BaseModel):
    """Model for updating an existing document."""
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    change_summary: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for document queries."""
    id: int
    title: str
    document_type: DocumentType
    status: DocumentStatus
    metadata: Dict[str, Any]
    version: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    indexed_at: Optional[datetime]
    uploaded_by: Optional[str]
    
    class Config:
        from_attributes = True


class DocumentVersionResponse(BaseModel):
    """Response model for document version queries."""
    id: int
    document_id: int
    version_number: int
    content_hash: str
    metadata: Dict[str, Any]
    change_summary: Optional[str]
    created_at: datetime
    created_by: Optional[str]
    
    class Config:
        from_attributes = True


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str
    document_types: Optional[List[DocumentType]] = None
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True


class DocumentSearchResult(BaseModel):
    """Single search result."""
    document: DocumentResponse
    similarity_score: float
    relevant_chunks: List[str] = Field(default_factory=list)


class DocumentSearchResponse(BaseModel):
    """Response model for document search."""
    query: str
    results: List[DocumentSearchResult]
    total_found: int
    processing_time_ms: float