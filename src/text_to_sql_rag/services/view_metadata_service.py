"""
Service for managing view metadata from MongoDB.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from pymongo import MongoClient
import structlog
from cachetools import TTLCache

from ..models.view_metadata_models import (
    ViewDomainMapping, 
    ViewDependency, 
    ViewQueryPattern,
    ViewMetadataConfig,
    ViewUsageStats
)
from ..config.settings import settings

logger = structlog.get_logger(__name__)


class ViewMetadataService:
    """Service for managing view metadata from MongoDB."""
    
    def __init__(self, mongo_client: Optional[MongoClient] = None):
        """
        Initialize view metadata service.
        
        Args:
            mongo_client: MongoDB client instance. If None, creates new client.
        """
        self.mongo_client = mongo_client or MongoClient(settings.mongodb.connection_string)
        self.db = self.mongo_client[settings.mongodb.database_name]
        
        # Collections
        self.view_mappings_collection = self.db.view_domain_mappings
        self.view_dependencies_collection = self.db.view_dependencies
        self.query_patterns_collection = self.db.view_query_patterns
        self.metadata_config_collection = self.db.view_metadata_config
        self.usage_stats_collection = self.db.view_usage_stats
        
        # Cache for performance
        self._mapping_cache: TTLCache = TTLCache(maxsize=1000, ttl=1800)  # 30 min TTL
        self._dependency_cache: TTLCache = TTLCache(maxsize=500, ttl=1800)
        self._pattern_cache: TTLCache = TTLCache(maxsize=200, ttl=1800)
        self._last_cache_refresh = None
        
        # Fallback hardcoded mappings (your existing ones)
        self._fallback_mappings = {
            "V_USER_METRICS": ["USER", "SYSTEM"],
            "V_DEAL_SUMMARY": ["DEAL", "ISSUER"],
            "V_TERMSHEET": ["DEAL", "TRANCHE", "ISSUER"],
            "V_TRANCHE_METRICS": ["TRANCHE", "DEAL"],
            "V_TRANCHE_PRICING": ["TRANCHE", "DEAL"],
            "V_ORDER_ALLOCATION": ["ORDER", "INVESTOR", "TRANCHE"],
            "V_SYNDICATE_PARTICIPATION": ["SYNDICATE", "TRANCHE"],
            "V_INVESTOR_PORTFOLIO": ["INVESTOR", "ORDER"],
            "V_TRADE_EXECUTION": ["TRADES", "ORDER", "INVESTOR"],
            "V_TRANCHE_INSTRUMENTS": ["TRANCHE", "DEAL", "SYNDICATE"],
            "V_ORDER_DETAILS": ["ORDER", "INVESTOR", "TRANCHE"],
            "V_ALLOCATION_SUMMARY": ["ORDER", "SYNDICATE", "INVESTOR"],
            "V_TRADE_SETTLEMENT": ["TRADES", "ORDER", "SYNDICATE"],
            "V_DEALER_TRADES": ["TRADES", "ORDER"]
        }
        
        # Initialize config
        self._config = self._get_or_create_config()
        
        logger.info("ViewMetadataService initialized", 
                   cache_ttl=1800, 
                   fallback_enabled=self._config.fallback_to_hardcoded)
    
    def get_view_domain_mappings(self, use_cache: bool = True) -> Dict[str, List[str]]:
        """
        Get all view to domain mappings.
        
        Args:
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping view names to business domain lists
        """
        cache_key = "all_mappings"
        
        # Check cache first
        if use_cache and cache_key in self._mapping_cache:
            if self._is_cache_fresh():
                return self._mapping_cache[cache_key]
        
        try:
            # Query MongoDB
            cursor = self.view_mappings_collection.find({})
            mappings = {}
            
            for doc in cursor:
                mapping = ViewDomainMapping(**doc)
                mappings[mapping.view_name] = mapping.business_domains
            
            if mappings:
                self._mapping_cache[cache_key] = mappings
                self._last_cache_refresh = datetime.utcnow()
                logger.info("Retrieved view mappings from MongoDB", count=len(mappings))
                return mappings
            else:
                logger.warning("No view mappings found in MongoDB, using fallback")
                return self._get_fallback_mappings()
                
        except Exception as e:
            logger.error("Failed to retrieve view mappings from MongoDB", error=str(e))
            if self._config.fallback_to_hardcoded:
                logger.info("Using fallback hardcoded mappings")
                return self._get_fallback_mappings()
            else:
                raise
    
    def get_view_dependencies(self, view_name: str = None) -> Dict[str, List[str]]:
        """
        Get view dependencies.
        
        Args:
            view_name: Specific view name, or None for all dependencies
            
        Returns:
            Dictionary mapping primary views to supporting view lists
        """
        cache_key = f"deps_{view_name or 'all'}"
        
        if cache_key in self._dependency_cache and self._is_cache_fresh():
            return self._dependency_cache[cache_key]
        
        try:
            query = {"primary_view": view_name} if view_name else {}
            cursor = self.view_dependencies_collection.find(query)
            
            dependencies = {}
            for doc in cursor:
                dep = ViewDependency(**doc)
                dependencies[dep.primary_view] = dep.supporting_views
            
            self._dependency_cache[cache_key] = dependencies
            return dependencies
            
        except Exception as e:
            logger.error("Failed to retrieve view dependencies", error=str(e), view_name=view_name)
            return {}
    
    def get_view_priorities(self) -> Dict[str, int]:
        """
        Get view priorities for ranking.
        
        Returns:
            Dictionary mapping view names to priority scores
        """
        try:
            cursor = self.view_mappings_collection.find({}, {"view_name": 1, "priority_score": 1})
            priorities = {}
            
            for doc in cursor:
                priorities[doc["view_name"]] = doc.get("priority_score", 5)
            
            return priorities
            
        except Exception as e:
            logger.error("Failed to retrieve view priorities", error=str(e))
            return {}
    
    def get_query_patterns(self, view_name: str = None) -> List[ViewQueryPattern]:
        """
        Get query patterns for view selection.
        
        Args:
            view_name: Specific view name, or None for all patterns
            
        Returns:
            List of ViewQueryPattern objects
        """
        cache_key = f"patterns_{view_name or 'all'}"
        
        if cache_key in self._pattern_cache and self._is_cache_fresh():
            return self._pattern_cache[cache_key]
        
        try:
            query = {"view_name": view_name} if view_name else {}
            cursor = self.query_patterns_collection.find(query)
            
            patterns = [ViewQueryPattern(**doc) for doc in cursor]
            self._pattern_cache[cache_key] = patterns
            
            return patterns
            
        except Exception as e:
            logger.error("Failed to retrieve query patterns", error=str(e), view_name=view_name)
            return []
    
    def add_view_mapping(self, mapping: ViewDomainMapping) -> bool:
        """
        Add or update a view domain mapping.
        
        Args:
            mapping: ViewDomainMapping object
            
        Returns:
            True if successful
        """
        try:
            mapping.updated_at = datetime.utcnow()
            
            # Upsert the mapping
            self.view_mappings_collection.replace_one(
                {"view_name": mapping.view_name},
                mapping.dict(),
                upsert=True
            )
            
            # Clear cache
            self._clear_mapping_cache()
            
            logger.info("Added/updated view mapping", 
                       view_name=mapping.view_name,
                       domains=mapping.business_domains)
            return True
            
        except Exception as e:
            logger.error("Failed to add view mapping", error=str(e), view_name=mapping.view_name)
            return False
    
    def remove_view_mapping(self, view_name: str) -> bool:
        """
        Remove a view domain mapping.
        
        Args:
            view_name: Name of view to remove
            
        Returns:
            True if successful
        """
        try:
            result = self.view_mappings_collection.delete_one({"view_name": view_name})
            
            if result.deleted_count > 0:
                self._clear_mapping_cache()
                logger.info("Removed view mapping", view_name=view_name)
                return True
            else:
                logger.warning("View mapping not found", view_name=view_name)
                return False
                
        except Exception as e:
            logger.error("Failed to remove view mapping", error=str(e), view_name=view_name)
            return False
    
    def update_usage_stats(self, view_name: str, success: bool, confidence: float = None, 
                          response_time_ms: float = None, query: str = None):
        """
        Update usage statistics for a view.
        
        Args:
            view_name: Name of the view
            success: Whether the query was successful
            confidence: Confidence score if available
            response_time_ms: Response time in milliseconds
            query: The query that was executed
        """
        try:
            # Get current stats or create new
            current_stats = self.usage_stats_collection.find_one({"view_name": view_name})
            
            if current_stats:
                stats = ViewUsageStats(**current_stats)
            else:
                stats = ViewUsageStats(view_name=view_name)
            
            # Update counters
            stats.usage_count += 1
            if success:
                stats.success_count += 1
            
            # Update confidence (running average)
            if confidence is not None:
                if stats.average_confidence == 0:
                    stats.average_confidence = confidence
                else:
                    # Simple moving average
                    stats.average_confidence = (stats.average_confidence + confidence) / 2
            
            # Update response time (running average)
            if response_time_ms is not None:
                if stats.average_response_time_ms == 0:
                    stats.average_response_time_ms = response_time_ms
                else:
                    stats.average_response_time_ms = (stats.average_response_time_ms + response_time_ms) / 2
            
            # Update error rate
            stats.error_rate = ((stats.usage_count - stats.success_count) / stats.usage_count) * 100
            
            # Update recent usage
            stats.last_used = datetime.utcnow()
            if query and success:
                stats.recent_queries.append(query)
                # Keep only last 10 queries
                stats.recent_queries = stats.recent_queries[-10:]
            
            stats.updated_at = datetime.utcnow()
            
            # Save to database
            self.usage_stats_collection.replace_one(
                {"view_name": view_name},
                stats.dict(),
                upsert=True
            )
            
        except Exception as e:
            logger.error("Failed to update usage stats", error=str(e), view_name=view_name)
    
    def get_usage_stats(self, view_name: str = None) -> Dict[str, ViewUsageStats]:
        """
        Get usage statistics.
        
        Args:
            view_name: Specific view name, or None for all stats
            
        Returns:
            Dictionary mapping view names to usage stats
        """
        try:
            query = {"view_name": view_name} if view_name else {}
            cursor = self.usage_stats_collection.find(query)
            
            stats = {}
            for doc in cursor:
                stat = ViewUsageStats(**doc)
                stats[stat.view_name] = stat
            
            return stats
            
        except Exception as e:
            logger.error("Failed to retrieve usage stats", error=str(e))
            return {}
    
    def _get_or_create_config(self) -> ViewMetadataConfig:
        """Get or create default configuration."""
        try:
            config_doc = self.metadata_config_collection.find_one({"config_name": "default"})
            
            if config_doc:
                return ViewMetadataConfig(**config_doc)
            else:
                # Create default config
                default_config = ViewMetadataConfig(config_name="default")
                self.metadata_config_collection.insert_one(default_config.dict())
                logger.info("Created default view metadata configuration")
                return default_config
                
        except Exception as e:
            logger.error("Failed to get/create config, using defaults", error=str(e))
            return ViewMetadataConfig(config_name="default")
    
    def _get_fallback_mappings(self) -> Dict[str, List[str]]:
        """Get hardcoded fallback mappings."""
        return self._fallback_mappings.copy()
    
    def _is_cache_fresh(self) -> bool:
        """Check if cache is still fresh based on TTL."""
        if not self._last_cache_refresh:
            return False
        
        ttl_minutes = self._config.cache_ttl_minutes
        cache_expiry = self._last_cache_refresh + timedelta(minutes=ttl_minutes)
        
        return datetime.utcnow() < cache_expiry
    
    def _clear_mapping_cache(self):
        """Clear the mapping cache."""
        self._mapping_cache.clear()
        self._last_cache_refresh = None
    
    def refresh_cache(self):
        """Force refresh of all caches."""
        self._mapping_cache.clear()
        self._dependency_cache.clear()
        self._pattern_cache.clear()
        self._last_cache_refresh = None
        logger.info("Cleared all view metadata caches")
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("Closed MongoDB connection")