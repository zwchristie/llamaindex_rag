"""
Service for managing business domain metadata from MongoDB.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from pymongo import MongoClient
import structlog
from cachetools import TTLCache

from ..models.business_domain_models import (
    BusinessDomainDefinition,
    DomainTerminology,
    DomainDetectionRule,
    ViewClassificationRule,
    BusinessContextConfig
)
from ..config.settings import settings

logger = structlog.get_logger(__name__)


class BusinessDomainMetadataService:
    """Service for managing business domain metadata from MongoDB."""
    
    def __init__(self, mongo_client: Optional[MongoClient] = None):
        """Initialize business domain metadata service."""
        self.mongo_client = mongo_client or MongoClient(settings.mongodb.url)
        self.db = self.mongo_client[settings.mongodb.database]
        
        # Collections
        self.domains_collection = self.db.business_domains
        self.terminology_collection = self.db.domain_terminology
        self.detection_rules_collection = self.db.domain_detection_rules
        self.classification_rules_collection = self.db.view_classification_rules
        self.context_config_collection = self.db.business_context_config
        
        # Cache for performance
        self._domain_cache: TTLCache = TTLCache(maxsize=100, ttl=1800)  # 30 min TTL
        self._terminology_cache: TTLCache = TTLCache(maxsize=500, ttl=1800)
        self._rules_cache: TTLCache = TTLCache(maxsize=200, ttl=1800)
        self._last_cache_refresh = None
        
        # Initialize config
        self._config = self._get_or_create_config()
        
        logger.info("BusinessDomainMetadataService initialized")
    
    def get_all_business_domains(self) -> Dict[str, BusinessDomainDefinition]:
        """Get all business domain definitions."""
        cache_key = "all_domains"
        
        if cache_key in self._domain_cache and self._is_cache_fresh():
            return self._domain_cache[cache_key]
        
        try:
            cursor = self.domains_collection.find({"is_active": True})
            domains = {}
            
            for doc in cursor:
                domain = BusinessDomainDefinition(**doc)
                domains[domain.domain_name] = domain
            
            if domains:
                self._domain_cache[cache_key] = domains
                self._last_cache_refresh = datetime.utcnow()
                logger.info("Retrieved business domains from MongoDB", count=len(domains))
                return domains
            else:
                logger.error("No business domains found in MongoDB! Please run the discovery script.")
                return {}
                
        except Exception as e:
            logger.error("Failed to retrieve business domains from MongoDB", error=str(e))
            return {}
    
    def get_domain_terminology(self, domain_name: str = None) -> Dict[str, List[DomainTerminology]]:
        """Get domain terminology mappings."""
        cache_key = f"terminology_{domain_name or 'all'}"
        
        if cache_key in self._terminology_cache and self._is_cache_fresh():
            return self._terminology_cache[cache_key]
        
        try:
            query = {"domain_name": domain_name} if domain_name else {}
            cursor = self.terminology_collection.find(query)
            
            terminology = {}
            for doc in cursor:
                term = DomainTerminology(**doc)
                if term.domain_name not in terminology:
                    terminology[term.domain_name] = []
                terminology[term.domain_name].append(term)
            
            self._terminology_cache[cache_key] = terminology
            return terminology
            
        except Exception as e:
            logger.error("Failed to retrieve domain terminology", error=str(e))
            return {}
    
    def get_domain_detection_rules(self) -> List[DomainDetectionRule]:
        """Get domain detection rules."""
        cache_key = "detection_rules"
        
        if cache_key in self._rules_cache and self._is_cache_fresh():
            return self._rules_cache[cache_key]
        
        try:
            cursor = self.detection_rules_collection.find().sort("priority", -1)
            rules = [DomainDetectionRule(**doc) for doc in cursor]
            
            self._rules_cache[cache_key] = rules
            logger.info("Retrieved domain detection rules from MongoDB", count=len(rules))
            return rules
            
        except Exception as e:
            logger.error("Failed to retrieve domain detection rules", error=str(e))
            return []
    
    def get_view_classification_rules(self) -> List[ViewClassificationRule]:
        """Get view classification rules."""
        cache_key = "classification_rules"
        
        if cache_key in self._rules_cache and self._is_cache_fresh():
            return self._rules_cache[cache_key]
        
        try:
            cursor = self.classification_rules_collection.find({"is_active": True})
            rules = [ViewClassificationRule(**doc) for doc in cursor]
            
            self._rules_cache[cache_key] = rules
            logger.info("Retrieved view classification rules from MongoDB", count=len(rules))
            return rules
            
        except Exception as e:
            logger.error("Failed to retrieve view classification rules", error=str(e))
            return []
    
    def add_business_domain(self, domain: BusinessDomainDefinition) -> bool:
        """Add or update a business domain definition."""
        try:
            domain.updated_at = datetime.utcnow()
            
            self.domains_collection.replace_one(
                {"domain_name": domain.domain_name},
                domain.dict(),
                upsert=True
            )
            
            self._clear_domain_cache()
            logger.info("Added/updated business domain", domain_name=domain.domain_name)
            return True
            
        except Exception as e:
            logger.error("Failed to add business domain", error=str(e), domain_name=domain.domain_name)
            return False
    
    def add_domain_terminology(self, terminology: DomainTerminology) -> bool:
        """Add or update domain terminology."""
        try:
            terminology.updated_at = datetime.utcnow()
            
            # Create unique constraint based on domain_name and term_type
            self.terminology_collection.replace_one(
                {
                    "domain_name": terminology.domain_name,
                    "term_type": terminology.term_type
                },
                terminology.dict(),
                upsert=True
            )
            
            self._clear_terminology_cache()
            logger.info("Added/updated domain terminology", 
                       domain_name=terminology.domain_name,
                       term_type=terminology.term_type)
            return True
            
        except Exception as e:
            logger.error("Failed to add domain terminology", error=str(e))
            return False
    
    def add_detection_rule(self, rule: DomainDetectionRule) -> bool:
        """Add or update a domain detection rule."""
        try:
            rule.updated_at = datetime.utcnow()
            
            self.detection_rules_collection.replace_one(
                {"rule_name": rule.rule_name},
                rule.dict(),
                upsert=True
            )
            
            self._clear_rules_cache()
            logger.info("Added/updated detection rule", rule_name=rule.rule_name)
            return True
            
        except Exception as e:
            logger.error("Failed to add detection rule", error=str(e), rule_name=rule.rule_name)
            return False
    
    def add_classification_rule(self, rule: ViewClassificationRule) -> bool:
        """Add or update a view classification rule."""
        try:
            rule.updated_at = datetime.utcnow()
            
            self.classification_rules_collection.replace_one(
                {"rule_name": rule.rule_name},
                rule.dict(),
                upsert=True
            )
            
            self._clear_rules_cache()
            logger.info("Added/updated classification rule", rule_name=rule.rule_name)
            return True
            
        except Exception as e:
            logger.error("Failed to add classification rule", error=str(e), rule_name=rule.rule_name)
            return False
    
    def get_domain_hierarchy(self) -> Dict[str, Dict[str, Any]]:
        """Get domain hierarchy information."""
        domains = self.get_all_business_domains()
        
        hierarchy = {}
        for domain_name, domain_def in domains.items():
            hierarchy[domain_name] = {
                "description": domain_def.description or domain_def.summary,
                "parent_domains": domain_def.parent_domains,
                "child_domains": domain_def.child_domains,
                "key_concepts": domain_def.key_concepts,
                "business_rules": domain_def.business_rules,
                "typical_queries": domain_def.typical_queries,
                "domain_level": domain_def.domain_level,
                "relationship_type": domain_def.relationship_type
            }
        
        return hierarchy
    
    def get_domain_key_concepts(self, domain_name: str) -> List[str]:
        """Get key concepts for a specific domain."""
        domains = self.get_all_business_domains()
        domain = domains.get(domain_name)
        
        if domain:
            # Combine key concepts from domain definition and terminology
            concepts = domain.key_concepts.copy()
            
            # Add terminology
            terminology = self.get_domain_terminology(domain_name)
            for term_list in terminology.get(domain_name, []):
                concepts.extend(term_list.terms)
            
            return list(set(concepts))  # Remove duplicates
        
        return []
    
    def classify_view(self, view_name: str, view_description: str = "", domains: List[str] = None) -> str:
        """Classify a view as core, supporting, or utility."""
        rules = self.get_view_classification_rules()
        domains = domains or []
        
        best_classification = "core"  # Default
        best_score = 0
        
        for rule in rules:
            score = 0
            matches = 0
            
            # Check name patterns
            view_upper = view_name.upper()
            for pattern in rule.name_patterns:
                if pattern.upper() in view_upper:
                    score += 2
                    matches += 1
            
            # Check description patterns
            if view_description:
                desc_upper = view_description.upper()
                for pattern in rule.description_patterns:
                    if pattern.upper() in desc_upper:
                        score += 1
                        matches += 1
            
            # Check domain requirements
            if rule.domain_requirements and domains:
                domain_matches = len(set(rule.domain_requirements).intersection(domains))
                if domain_matches > 0:
                    score += domain_matches
                    matches += 1
            
            # Apply rule logic
            if rule.match_logic == "any" and matches > 0:
                score += rule.priority_boost
            elif rule.match_logic == "all" and matches == len(rule.name_patterns + rule.description_patterns + rule.domain_requirements):
                score += rule.priority_boost * 2
            
            # Update best classification
            if score > best_score:
                best_score = score
                best_classification = rule.classification_type
        
        return best_classification
    
    def _get_or_create_config(self) -> BusinessContextConfig:
        """Get or create default configuration."""
        try:
            config_doc = self.context_config_collection.find_one({"config_name": "default"})
            
            if config_doc:
                return BusinessContextConfig(**config_doc)
            else:
                default_config = BusinessContextConfig(config_name="default")
                self.context_config_collection.insert_one(default_config.dict())
                logger.info("Created default business context configuration")
                return default_config
                
        except Exception as e:
            logger.error("Failed to get/create config, using defaults", error=str(e))
            return BusinessContextConfig(config_name="default")
    
    def _is_cache_fresh(self) -> bool:
        """Check if cache is still fresh."""
        if not self._last_cache_refresh:
            return False
        
        ttl_minutes = self._config.cache_ttl_minutes if self._config.enable_caching else 0
        if ttl_minutes <= 0:
            return False
            
        cache_expiry = self._last_cache_refresh + timedelta(minutes=ttl_minutes)
        return datetime.utcnow() < cache_expiry
    
    def _clear_domain_cache(self):
        """Clear domain cache."""
        keys_to_clear = [key for key in self._domain_cache.keys() if "domain" in key]
        for key in keys_to_clear:
            self._domain_cache.pop(key, None)
        self._last_cache_refresh = None
    
    def _clear_terminology_cache(self):
        """Clear terminology cache."""
        self._terminology_cache.clear()
        self._last_cache_refresh = None
    
    def _clear_rules_cache(self):
        """Clear rules cache."""
        self._rules_cache.clear()
        self._last_cache_refresh = None
    
    def refresh_cache(self):
        """Force refresh of all caches."""
        self._domain_cache.clear()
        self._terminology_cache.clear()
        self._rules_cache.clear()
        self._last_cache_refresh = None
        logger.info("Cleared all business domain metadata caches")
    
    def create_indexes(self):
        """Create MongoDB indexes for performance."""
        try:
            # Business domains indexes
            self.domains_collection.create_index("domain_name", unique=True)
            self.domains_collection.create_index("is_active")
            self.domains_collection.create_index("parent_domains")
            
            # Terminology indexes
            self.terminology_collection.create_index([("domain_name", 1), ("term_type", 1)], unique=True)
            self.terminology_collection.create_index("terms")
            
            # Detection rules indexes
            self.detection_rules_collection.create_index("rule_name", unique=True)
            self.detection_rules_collection.create_index("priority")
            self.detection_rules_collection.create_index("target_domains")
            
            # Classification rules indexes
            self.classification_rules_collection.create_index("rule_name", unique=True)
            self.classification_rules_collection.create_index("is_active")
            
            logger.info("Created business domain metadata indexes")
            
        except Exception as e:
            logger.error("Error creating business domain indexes", error=str(e))
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Closed business domain metadata MongoDB connection")