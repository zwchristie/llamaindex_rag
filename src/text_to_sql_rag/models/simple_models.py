"""Simplified models for API requests and responses without database dependencies."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for 4-tier metadata architecture."""
    # Legacy types (being phased out)
    SCHEMA = "schema"  # Old monolithic schema files
    REPORT = "report"  # Report documentation and examples
    
    # New 4-tier types
    DDL = "ddl"                          # Core table/view structure (.sql files)
    COLUMN_DETAILS = "column_details"     # Detailed column metadata (.json) 
    LOOKUP_METADATA = "lookup_metadata"   # ID-name lookup mappings (.json)
    # REPORTS reused for query examples and business context


class DocumentMetadata(BaseModel):
    """Document metadata structure."""
    title: str
    description: Optional[str] = None
    document_type: DocumentType
    tags: List[str] = Field(default_factory=list)
    file_name: Optional[str] = None
    uploaded_at: Optional[datetime] = None


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    title: str
    document_type: DocumentType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str
    document_types: Optional[List[DocumentType]] = None
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True


class DocumentSearchResult(BaseModel):
    """Single search result."""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    document_type: Optional[str] = None


class DocumentSearchResponse(BaseModel):
    """Response model for document search."""
    query: str
    results: List[DocumentSearchResult]
    total_found: int
    processing_time_ms: Optional[float] = None


class SQLGenerationRequest(BaseModel):
    """Request model for SQL generation."""
    query: str
    session_id: Optional[str] = None
    use_hybrid_retrieval: bool = True
    auto_execute: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)


class SQLGenerationResponse(BaseModel):
    """Response model for SQL generation."""
    natural_query: str
    sql_query: str
    explanation: str
    confidence: float
    tables_used: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    context_quality: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    session_id: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    auto_executed: bool = False


class QueryExecutionRequest(BaseModel):
    """Request model for query execution."""
    sql_query: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryValidationRequest(BaseModel):
    """Request model for query validation."""
    sql_query: str


class QueryValidationResponse(BaseModel):
    """Response model for query validation."""
    valid: bool
    sql_query: str
    error: Optional[str] = None
    suggestions: Optional[str] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class QueryExplanationRequest(BaseModel):
    """Request model for query explanation."""
    sql_query: str


class QueryExplanationResponse(BaseModel):
    """Response model for query explanation."""
    sql_query: str
    explanation: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class SessionRequest(BaseModel):
    """Request model for session creation."""
    initial_query: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Response model for session information."""
    session_id: str
    initial_query: str
    user_id: Optional[str] = None
    context: Dict[str, Any]
    created_at: datetime
    status: str = "active"


class InteractionRequest(BaseModel):
    """Request model for adding interaction to session."""
    interaction_type: str
    user_input: Optional[str] = None
    system_response: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HumanInterventionRequest(BaseModel):
    """Request for human intervention."""
    session_id: str
    message: str
    suggested_actions: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    timeout_minutes: int = Field(default=30, ge=1, le=120)


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    database: Optional[str] = None
    vector_store: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)