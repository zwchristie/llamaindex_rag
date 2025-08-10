"""
New simplified view metadata models - one document per view.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ViewColumn(BaseModel):
    """Column information for a database view."""
    
    name: str = Field(..., description="Column name")
    type: str = Field(..., description="Column data type")
    notNull: bool = Field(False, description="Whether column allows NULL values")
    description: Optional[str] = Field(None, description="Column description")


class ViewJoin(BaseModel):
    """Join relationship information."""
    
    table_name: str = Field(..., description="Related table/view name")
    join_type: str = Field("INNER", description="Type of join (INNER, LEFT, RIGHT, etc.)")
    join_condition: str = Field(..., description="Join condition")
    description: Optional[str] = Field(None, description="Join description")


class ViewMetadata(BaseModel):
    """Complete metadata for a single database view."""
    
    # Core identification
    view_name: str = Field(..., description="Database view name")
    view_type: str = Field("CORE", description="View type: CORE or SUPPORTING")
    schema_name: Optional[str] = Field(None, description="Database schema name")
    
    # Descriptive information
    description: str = Field("", description="View description")
    use_cases: str = Field("", description="Common use cases for this view")
    
    # Structure
    columns: List[ViewColumn] = Field(..., description="Column definitions")
    joins: List[ViewJoin] = Field(default_factory=list, description="Join relationships")
    
    # SQL information
    view_sql: Optional[str] = Field(None, description="View definition SQL")
    sample_sql: Optional[str] = Field(None, description="Sample query using this view")
    example_query: Optional[str] = Field(None, description="Example query string")
    
    # Sample data and context
    data_returned: Optional[str] = Field(None, description="Description of data returned")
    sample_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sample data rows")
    
    # Concatenated text for embedding (computed field)
    full_text: Optional[str] = Field(None, description="Concatenated text for embedding")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def generate_full_text(self) -> str:
        """Generate concatenated text for embedding."""
        parts = [
            f"View: {self.view_name}",
            f"Type: {self.view_type}",
        ]
        
        if self.schema_name:
            parts.append(f"Schema: {self.schema_name}")
            
        if self.description:
            parts.append(f"Description: {self.description}")
            
        if self.use_cases:
            parts.append(f"Use Cases: {self.use_cases}")
            
        # Add columns
        if self.columns:
            col_info = []
            for col in self.columns:
                col_text = f"{col.name} ({col.type})"
                if col.notNull:
                    col_text += " NOT NULL"
                if col.description:
                    col_text += f": {col.description}"
                col_info.append(col_text)
            parts.append(f"Columns: {', '.join(col_info)}")
            
        # Add joins
        if self.joins:
            join_info = []
            for join in self.joins:
                join_text = f"{join.join_type} JOIN {join.table_name} ON {join.join_condition}"
                if join.description:
                    join_text += f" ({join.description})"
                join_info.append(join_text)
            parts.append(f"Joins: {'; '.join(join_info)}")
            
        if self.sample_sql:
            parts.append(f"Sample SQL: {self.sample_sql}")
            
        if self.data_returned:
            parts.append(f"Data Returned: {self.data_returned}")
            
        return "\n".join(parts)
    
    class Config:
        collection = "view_metadata"


class ViewEmbedding(BaseModel):
    """Embedding document for OpenSearch storage."""
    
    view_name: str = Field(..., description="Database view name")
    full_text: str = Field(..., description="Concatenated text content")
    embedding: List[float] = Field(..., description="Vector embedding")
    metadata: ViewMetadata = Field(..., description="Complete view metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionState(BaseModel):
    """Agent session state for resumable flows."""
    
    session_id: str = Field(..., description="Unique session identifier")
    current_step: str = Field(..., description="Current processing step")
    user_query: str = Field(..., description="Original user query")
    
    # Retrieved context
    retrieved_views: List[ViewMetadata] = Field(default_factory=list)
    selected_views: List[str] = Field(default_factory=list)
    
    # Generated content
    generated_sql: Optional[str] = Field(None, description="Generated SQL query")
    sql_explanation: Optional[str] = Field(None, description="SQL explanation")
    
    # HITL state
    hitl_request_id: Optional[str] = Field(None, description="HITL request ID if awaiting approval")
    hitl_status: Optional[str] = Field(None, description="HITL status: pending|approved|rejected")
    
    # Final result
    query_result: Optional[Any] = Field(None, description="Final query result")
    formatted_response: Optional[str] = Field(None, description="Formatted response")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "session_states"


class ReportMetadata(BaseModel):
    """Metadata for database reports and report definitions."""
    
    # Core identification
    report_name: str = Field(..., description="Report name or identifier")
    view_name: Optional[str] = Field(None, description="Underlying view name if applicable")
    report_type: str = Field("STANDARD", description="Report type: STANDARD, DASHBOARD, SUMMARY")
    
    # Descriptive information
    report_description: str = Field("", description="Report description and purpose")
    data_returned: Optional[str] = Field(None, description="Description of data returned by report")
    use_cases: str = Field("", description="Common use cases for this report")
    
    # SQL information
    example_sql: Optional[str] = Field(None, description="Example SQL for this report")
    filters: Optional[List[str]] = Field(default_factory=list, description="Common filter conditions")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def generate_full_text(self) -> str:
        """Generate concatenated text for embedding."""
        parts = [
            f"Report: {self.report_name}",
            f"Type: {self.report_type}",
        ]
        
        if self.view_name:
            parts.append(f"View: {self.view_name}")
            
        if self.report_description:
            parts.append(f"Description: {self.report_description}")
            
        if self.use_cases:
            parts.append(f"Use Cases: {self.use_cases}")
            
        if self.data_returned:
            parts.append(f"Data Returned: {self.data_returned}")
            
        if self.example_sql:
            parts.append(f"Example SQL: {self.example_sql}")
            
        if self.filters:
            parts.append(f"Common Filters: {', '.join(self.filters)}")
            
        return "\n".join(parts)


class LookupValue(BaseModel):
    """Individual lookup value entry."""
    
    id: int = Field(..., description="Lookup value ID")
    name: str = Field(..., description="Lookup value name")
    code: str = Field(..., description="Lookup code or abbreviation")
    description: str = Field("", description="Description of the lookup value")


class LookupMetadata(BaseModel):
    """Metadata for lookup tables and reference data."""
    
    # Core identification
    lookup_name: str = Field(..., description="Lookup table or reference name")
    lookup_type: str = Field("REFERENCE", description="Lookup type: REFERENCE, STATUS, CATEGORY")
    
    # Descriptive information
    description: str = Field("", description="Description of the lookup table")
    use_cases: str = Field("", description="How this lookup is typically used")
    
    # Lookup values
    values: List[LookupValue] = Field(..., description="List of lookup values")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def generate_full_text(self) -> str:
        """Generate concatenated text for embedding."""
        parts = [
            f"Lookup: {self.lookup_name}",
            f"Type: {self.lookup_type}",
        ]
        
        if self.description:
            parts.append(f"Description: {self.description}")
            
        if self.use_cases:
            parts.append(f"Use Cases: {self.use_cases}")
            
        # Add values summary
        if self.values:
            value_summary = []
            for value in self.values[:10]:  # Limit to first 10 for space
                value_text = f"{value.code} ({value.name})"
                if value.description:
                    value_text += f": {value.description}"
                value_summary.append(value_text)
            
            parts.append(f"Values: {'; '.join(value_summary)}")
            
            if len(self.values) > 10:
                parts.append(f"... and {len(self.values) - 10} more values")
        
        return "\n".join(parts)


class HITLRequest(BaseModel):
    """Human-in-the-Loop approval request."""
    
    request_id: str = Field(..., description="Unique request identifier")
    session_id: str = Field(..., description="Associated session ID")
    
    # Request details
    user_query: str = Field(..., description="Original user query")
    generated_sql: str = Field(..., description="Generated SQL requiring approval")
    sql_explanation: str = Field(..., description="Explanation of the SQL")
    selected_views: List[str] = Field(..., description="Views used in query")
    
    # Status and resolution
    status: str = Field("pending", description="Status: pending|approved|rejected")
    reviewer_notes: Optional[str] = Field(None, description="Notes from reviewer")
    resolution_reason: Optional[str] = Field(None, description="Reason for approval/rejection")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(None, description="When request was resolved")
    expires_at: datetime = Field(..., description="When request expires")
    
    class Config:
        collection = "hitl_requests"


class ReportMetadata(BaseModel):
    """Report example metadata (unchanged from current structure)."""
    
    report_name: str = Field(..., description="Report identifier")
    description: str = Field(..., description="Report description")
    sample_queries: List[str] = Field(..., description="Example queries for this report")
    expected_columns: List[str] = Field(..., description="Expected output columns")
    use_cases: List[str] = Field(default_factory=list, description="Use cases")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "report_metadata"


class LookupMetadata(BaseModel):
    """Lookup information metadata (unchanged from current structure)."""
    
    lookup_name: str = Field(..., description="Lookup identifier")
    description: str = Field(..., description="Lookup description")
    values: List[Dict[str, Any]] = Field(..., description="Lookup values")
    context: str = Field("", description="Context for when to use this lookup")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "lookup_metadata"