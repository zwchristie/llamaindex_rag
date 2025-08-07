"""
Pydantic models for view metadata storage in MongoDB.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ViewDomainMapping(BaseModel):
    """Model for view to business domain mappings."""
    
    view_name: str = Field(..., description="Database view name (e.g., V_TERMSHEET)")
    business_domains: List[str] = Field(..., description="List of business domains this view covers")
    view_type: str = Field("core", description="View type: core or supporting")
    priority_score: int = Field(5, description="Priority score 1-10 for ranking within domain")
    description: str = Field("", description="Brief description of what this view contains")
    key_entities: List[str] = Field(default_factory=list, description="Primary entities/tables this view uses")
    query_patterns: List[str] = Field(default_factory=list, description="Common query patterns this view supports")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field("system", description="Who created this mapping")
    
    class Config:
        collection = "view_domain_mappings"


class ViewDependency(BaseModel):
    """Model for view dependencies."""
    
    primary_view: str = Field(..., description="Main view name")
    supporting_views: List[str] = Field(..., description="Views that support/enhance the primary view")
    dependency_type: str = Field("enhancement", description="Type of dependency: enhancement, required, optional")
    description: str = Field("", description="Description of how supporting views enhance primary view")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "view_dependencies"


class ViewQueryPattern(BaseModel):
    """Model for view-specific query patterns and routing rules."""
    
    view_name: str = Field(..., description="Database view name")
    query_keywords: List[str] = Field(..., description="Keywords that indicate this view should be used")
    query_patterns: List[str] = Field(..., description="Regex patterns for query matching")
    business_context_indicators: List[str] = Field(..., description="Business context phrases that suggest this view")
    sample_queries: List[str] = Field(default_factory=list, description="Example queries that use this view")
    
    # Scoring weights
    keyword_weight: float = Field(1.0, description="Weight for keyword matching")
    pattern_weight: float = Field(1.5, description="Weight for pattern matching")
    context_weight: float = Field(2.0, description="Weight for business context matching")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "view_query_patterns"


class ViewMetadataConfig(BaseModel):
    """Configuration for view metadata management."""
    
    config_name: str = Field(..., description="Configuration name (e.g., 'default', 'production')")
    auto_discovery_enabled: bool = Field(True, description="Whether to auto-discover new views")
    cache_ttl_minutes: int = Field(30, description="Cache TTL for view mappings in minutes")
    fallback_to_hardcoded: bool = Field(True, description="Fall back to hardcoded mappings if DB fails")
    
    # Default scoring
    default_core_priority: int = Field(7, description="Default priority for core views")
    default_supporting_priority: int = Field(4, description="Default priority for supporting views")
    
    # Update settings
    last_sync_timestamp: Optional[datetime] = Field(None, description="Last successful sync with database")
    sync_frequency_minutes: int = Field(60, description="How often to sync metadata from database")
    
    class Config:
        collection = "view_metadata_config"


class ViewUsageStats(BaseModel):
    """Model for tracking view usage statistics."""
    
    view_name: str = Field(..., description="Database view name")
    usage_count: int = Field(0, description="Number of times this view was selected")
    success_count: int = Field(0, description="Number of successful queries using this view")
    average_confidence: float = Field(0.0, description="Average confidence score for this view")
    
    # Recent usage
    last_used: Optional[datetime] = Field(None, description="Last time this view was used")
    recent_queries: List[str] = Field(default_factory=list, description="Recent successful queries")
    
    # Performance metrics
    average_response_time_ms: float = Field(0.0, description="Average response time in milliseconds")
    error_rate: float = Field(0.0, description="Error rate percentage")
    
    # Update tracking
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "view_usage_stats"