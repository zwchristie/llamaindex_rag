"""
Pydantic models for business domain metadata storage in MongoDB.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class BusinessDomainDefinition(BaseModel):
    """Model for business domain definitions."""
    
    domain_id: str = Field(..., description="Unique domain identifier (e.g., 'issuer')")
    domain_name: str = Field(..., description="Display name (e.g., 'ISSUER')")
    summary: str = Field(..., description="Brief description of the domain")
    description: str = Field("", description="Detailed description")
    
    # Hierarchy relationships
    parent_domains: List[str] = Field(default_factory=list, description="Parent domain names")
    child_domains: List[str] = Field(default_factory=list, description="Child domain names")
    
    # Domain characteristics
    key_concepts: List[str] = Field(default_factory=list, description="Key terms and concepts")
    business_rules: List[str] = Field(default_factory=list, description="Business rules for this domain")
    typical_queries: List[str] = Field(default_factory=list, description="Typical query patterns")
    
    # Classification
    domain_level: int = Field(1, description="Hierarchy level (1=root, higher=deeper)")
    relationship_type: str = Field("standalone", description="root_entity, intermediate_entity, leaf_entity")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field("system", description="Who created this domain definition")
    is_active: bool = Field(True, description="Whether this domain is active")
    
    class Config:
        collection = "business_domains"


class DomainTerminology(BaseModel):
    """Model for domain-specific terminology and keywords."""
    
    domain_name: str = Field(..., description="Domain this terminology belongs to")
    term_type: str = Field(..., description="Type: primary, synonym, related, technical")
    terms: List[str] = Field(..., description="List of terms/keywords")
    weight: float = Field(1.0, description="Importance weight for query matching")
    
    # Context
    context_phrases: List[str] = Field(default_factory=list, description="Contextual phrases")
    exclusion_terms: List[str] = Field(default_factory=list, description="Terms that exclude this domain")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "domain_terminology"


class DomainDetectionRule(BaseModel):
    """Model for domain detection rules and patterns."""
    
    rule_name: str = Field(..., description="Unique rule identifier")
    target_domains: List[str] = Field(..., description="Domains this rule detects")
    
    # Detection patterns
    keyword_patterns: List[str] = Field(default_factory=list, description="Keywords that trigger this rule")
    phrase_patterns: List[str] = Field(default_factory=list, description="Phrase patterns (can include regex)")
    context_requirements: List[str] = Field(default_factory=list, description="Additional context requirements")
    
    # Rule logic
    match_type: str = Field("any", description="any, all, weighted")
    confidence_weight: float = Field(1.0, description="Weight for confidence scoring")
    minimum_matches: int = Field(1, description="Minimum matches required")
    
    # Rule metadata
    priority: int = Field(5, description="Rule priority (1-10)")
    is_fallback: bool = Field(False, description="Whether this is a fallback rule")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "domain_detection_rules"


class ViewClassificationRule(BaseModel):
    """Model for view classification rules."""
    
    rule_name: str = Field(..., description="Rule identifier")
    classification_type: str = Field(..., description="core, supporting, utility")
    
    # Pattern matching
    name_patterns: List[str] = Field(default_factory=list, description="View name patterns")
    description_patterns: List[str] = Field(default_factory=list, description="Description patterns")
    domain_requirements: List[str] = Field(default_factory=list, description="Required domains")
    
    # Classification criteria
    priority_boost: int = Field(0, description="Priority adjustment for matching views")
    match_logic: str = Field("any", description="any, all, weighted")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(True)
    
    class Config:
        collection = "view_classification_rules"


class BusinessContextConfig(BaseModel):
    """Configuration for business context and domain management."""
    
    config_name: str = Field(..., description="Configuration name")
    
    # Domain settings
    max_domains_per_query: int = Field(5, description="Maximum domains to identify per query")
    domain_confidence_threshold: float = Field(0.3, description="Minimum confidence for domain detection")
    enable_domain_expansion: bool = Field(True, description="Whether to expand to related domains")
    
    # Terminology settings
    enable_query_enhancement: bool = Field(True, description="Whether to enhance queries with domain terms")
    terminology_weight: float = Field(1.0, description="Weight for terminology matching")
    max_enhancement_terms: int = Field(10, description="Maximum terms to add to query")
    
    # Hierarchy settings
    respect_hierarchy: bool = Field(True, description="Whether to respect domain hierarchy")
    include_parent_domains: bool = Field(True, description="Include parent domains in context")
    include_child_domains: bool = Field(False, description="Include child domains in context")
    
    # Cache settings
    cache_ttl_minutes: int = Field(30, description="Cache TTL in minutes")
    enable_caching: bool = Field(True, description="Whether to enable caching")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        collection = "business_context_config"