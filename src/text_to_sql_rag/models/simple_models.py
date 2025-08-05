"""Simplified models for API requests and responses without database dependencies."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for business domain-first cascading RAG architecture."""
    
    # Tier 1: Business Intelligence (Domain-First)
    BUSINESS_DOMAIN = "business_domain"      # Entity hierarchy & relationships
    
    # Tier 2: Data Views (Domain-Filtered)
    CORE_VIEW = "core_view"                  # Primary business objects
    SUPPORTING_VIEW = "supporting_view"       # Extended view data with dependencies
    
    # Tier 3: Query Patterns (Context-Aware)
    REPORT = "report"                        # Query examples & use cases
    
    # Supporting Types
    DDL = "ddl"                             # View DDL statements
    LOOKUP_METADATA = "lookup_metadata"      # ID-name mappings
    
    # Legacy types (being phased out)
    SCHEMA = "schema"                       # Old monolithic schema files
    COLUMN_DETAILS = "column_details"       # Old detailed column metadata


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


# New Business Domain-First Metadata Models

class BusinessDomainMetadata(BaseModel):
    """Business domain entity metadata for cascading RAG."""
    domain_name: str = Field(..., description="Business domain name (e.g., TRANCHE, ORDER)")
    description: str = Field(..., description="Business definition of the domain")
    parent_domains: List[str] = Field(default_factory=list, description="Parent entities in hierarchy")
    child_domains: List[str] = Field(default_factory=list, description="Child entities in hierarchy") 
    key_concepts: List[str] = Field(default_factory=list, description="Domain-specific terminology")
    related_views: List[str] = Field(default_factory=list, description="Views assigned to this domain")
    business_rules: List[str] = Field(default_factory=list, description="Domain-specific business rules")


class CoreViewMetadata(BaseModel):
    """Core view metadata for primary business objects."""
    view_name: str = Field(..., description="View name")
    view_type: str = Field(default="Core view", description="View classification")
    description: str = Field(..., description="View description and purpose")
    business_domains: List[str] = Field(default_factory=list, description="Assigned business domains")
    data_returned: str = Field(..., description="Description of data returned by view")
    use_cases: List[str] = Field(default_factory=list, description="Common use cases for this view")
    example_query: Optional[str] = Field(None, description="Example SQL query")
    view_sql: Optional[str] = Field(None, description="View creation SQL")
    financial_context: Optional[str] = Field(None, description="Financial domain context")


class SupportingViewMetadata(BaseModel):
    """Supporting view metadata with dependencies."""
    view_name: str = Field(..., description="View name")
    view_type: str = Field(default="Supporting view", description="View classification")
    description: str = Field(..., description="View description and purpose")
    business_domains: List[str] = Field(default_factory=list, description="Assigned business domains")
    views_supported: List[str] = Field(default_factory=list, description="Core views this supports")
    data_returned: str = Field(..., description="Description of data returned by view")
    use_cases: List[str] = Field(default_factory=list, description="Common use cases for this view")
    example_query: Optional[str] = Field(None, description="Example SQL query")
    view_sql: Optional[str] = Field(None, description="View creation SQL")
    enhancement_provided: Optional[str] = Field(None, description="How this enhances core views")


class ReportMetadata(BaseModel):
    """Report metadata for query patterns and examples."""
    name: str = Field(..., description="Report name")
    report_description: str = Field(..., description="Report purpose and description")
    business_domains: List[str] = Field(default_factory=list, description="Relevant business domains")
    related_views: List[str] = Field(default_factory=list, description="Views used in this report")
    use_cases: List[str] = Field(default_factory=list, description="Report use cases")
    data_returned: Optional[str] = Field(None, description="Description of data returned")
    example_sql: Optional[str] = Field(None, description="Example SQL query")
    view_name: Optional[str] = Field(None, description="Associated view if exists")
    view_sql: Optional[str] = Field(None, description="Associated view SQL if exists")
    query_patterns: List[str] = Field(default_factory=list, description="Common query patterns")


class DomainContext(BaseModel):
    """Result of business domain identification stage."""
    identified_domains: List[str] = Field(..., description="Identified business domains")
    enhanced_query: str = Field(..., description="Query enhanced with domain terminology")
    business_context: str = Field(..., description="Relevant business context")
    confidence: float = Field(..., description="Confidence in domain identification")
    domain_relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Domain hierarchy context")


class ViewContext(BaseModel):
    """Result of domain-filtered view selection stage."""
    core_views: List[str] = Field(default_factory=list, description="Selected core views")
    supporting_views: List[str] = Field(default_factory=list, description="Selected supporting views")
    view_dependencies: Dict[str, List[str]] = Field(default_factory=dict, description="View dependency mapping")
    business_domains: List[str] = Field(default_factory=list, description="Domains these views belong to")
    selection_reasoning: str = Field(..., description="Why these views were selected")


class ReportContext(BaseModel):
    """Result of report pattern extraction stage."""
    relevant_reports: List[str] = Field(default_factory=list, description="Selected report files")
    sql_patterns: List[str] = Field(default_factory=list, description="Extracted SQL patterns")
    use_case_matches: List[str] = Field(default_factory=list, description="Matching use cases")
    query_examples: List[str] = Field(default_factory=list, description="Similar query examples")
    pattern_confidence: float = Field(..., description="Confidence in pattern matching")