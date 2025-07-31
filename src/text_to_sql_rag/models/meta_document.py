"""Models for metadata documents (schema and reports).

NOTE: SchemaMetadata is primarily used for legacy schema files.
New hierarchical architecture uses separate DDL, BUSINESS_DESC, BUSINESS_RULES, 
and COLUMN_DETAILS document types with simpler JSON structures.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

from .simple_models import DocumentType


class ColumnInfo(BaseModel):
    """Information about a database column."""
    name: str
    type: str
    key: Optional[str] = None  # PRIMARY KEY, FOREIGN KEY, UNIQUE, etc.
    example_values: List[str] = Field(default_factory=list)
    nullable: bool = True


class TableModel(BaseModel):
    """Information about a database table."""
    table_name: str
    columns: List[ColumnInfo]


class ViewInfo(BaseModel):
    """Information about a database view."""
    view_name: str
    query: str
    columns: List[ColumnInfo]


class RelationshipInfo(BaseModel):
    """Information about table relationships."""
    relationship_name: str
    tables: List[str]
    example_sql: str
    type: str  # one-to-one, one-to-many, many-to-one, many-to-many
    description: Optional[str] = None


class SchemaMetadata(BaseModel):
    """Schema metadata document structure."""
    catalog: str
    schema_name: str = Field(..., alias="schema")
    models: List[TableModel]
    views: List[ViewInfo] = Field(default_factory=list)
    relationships: List[RelationshipInfo] = Field(default_factory=list)
    
    @validator('catalog', 'schema_name')
    def validate_names(cls, v):
        if not v or not v.strip():
            raise ValueError('Catalog and schema names cannot be empty')
        return v.strip()
    
    class Config:
        allow_population_by_field_name = True


class ReportMetadata(BaseModel):
    """Report metadata extracted from text files."""
    catalog: str
    report_name: str
    description: str
    sql_query: str
    data_returned: str
    use_cases: Optional[str] = None
    
    @validator('catalog', 'report_name')
    def validate_names(cls, v):
        if not v or not v.strip():
            raise ValueError('Catalog and report names cannot be empty')
        return v.strip()


class MetaDocument(BaseModel):
    """Base metadata document model."""
    file_path: str
    content: str
    content_hash: str
    document_type: DocumentType
    catalog: str
    schema_name: str
    metadata: Union[SchemaMetadata, ReportMetadata]
    last_modified: datetime
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentSyncResult(BaseModel):
    """Result of document synchronization operation."""
    file_path: str
    action: str  # "created", "updated", "skipped", "error"
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None


class SyncSummary(BaseModel):
    """Summary of document synchronization process."""
    total_files_processed: int
    mongodb_operations: Dict[str, int] = Field(default_factory=dict)  # created, updated, skipped, errors
    vector_store_operations: Dict[str, int] = Field(default_factory=dict)  # added, updated, skipped, errors
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }