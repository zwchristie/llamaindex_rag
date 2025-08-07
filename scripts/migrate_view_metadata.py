#!/usr/bin/env python3
"""
Migration script to populate MongoDB with view metadata from hardcoded mappings.
This script initializes the view domain mappings, dependencies, and metadata in MongoDB.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.models.view_metadata_models import (
    ViewDomainMapping,
    ViewDependency,
    ViewQueryPattern,
    ViewMetadataConfig,
    ViewUsageStats
)
from text_to_sql_rag.services.view_metadata_service import ViewMetadataService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViewMetadataMigrator:
    """Migration utility for view metadata."""
    
    def __init__(self):
        """Initialize migrator."""
        self.metadata_service = ViewMetadataService()
        logger.info("ViewMetadataMigrator initialized")
    
    def migrate_view_domain_mappings(self):
        """Migrate hardcoded view domain mappings to MongoDB."""
        logger.info("Starting view domain mappings migration...")
        
        # Hardcoded mappings from ViewSelectionService
        mappings_data = {
            # Core Views - Primary business entity views
            "V_USER_METRICS": {
                "domains": ["USER", "SYSTEM"],
                "type": "core",
                "priority": 6,
                "description": "User activity and system metrics",
                "entities": ["users", "system_metrics"],
                "patterns": ["user", "metrics", "activity", "system"]
            },
            "V_DEAL_SUMMARY": {
                "domains": ["DEAL", "ISSUER"],
                "type": "core",
                "priority": 8,
                "description": "High-level deal summary and issuer information",
                "entities": ["deals", "issuers"],
                "patterns": ["deal", "summary", "issuer", "overview"]
            },
            "V_TERMSHEET": {
                "domains": ["DEAL", "TRANCHE", "ISSUER"],
                "type": "core",
                "priority": 10,  # Highest priority for deal/tranche queries
                "description": "Primary deal and tranche master data with detailed terms",
                "entities": ["deals", "tranches", "issuers", "termsheets"],
                "patterns": ["deal", "tranche", "termsheet", "terms", "announced", "status", "fixed income"]
            },
            "V_TRANCHE_METRICS": {
                "domains": ["TRANCHE", "DEAL"],
                "type": "core",
                "priority": 8,
                "description": "Tranche performance metrics and analytics",
                "entities": ["tranches", "deals"],
                "patterns": ["tranche", "metrics", "performance", "analytics"]
            },
            "V_TRANCHE_PRICING": {
                "domains": ["TRANCHE", "DEAL"],
                "type": "core",
                "priority": 7,
                "description": "Tranche pricing, yield, and spread information",
                "entities": ["tranches", "pricing"],
                "patterns": ["tranche", "pricing", "yield", "spread", "rates"]
            },
            "V_ORDER_ALLOCATION": {
                "domains": ["ORDER", "INVESTOR", "TRANCHE"],
                "type": "core",
                "priority": 7,
                "description": "Order allocation and investor assignment data",
                "entities": ["orders", "investors", "allocations"],
                "patterns": ["order", "allocation", "investor", "assignment"]
            },
            "V_SYNDICATE_PARTICIPATION": {
                "domains": ["SYNDICATE", "TRANCHE"],
                "type": "core",
                "priority": 6,
                "description": "Syndicate bank participation and roles",
                "entities": ["syndicate", "banks", "participation"],
                "patterns": ["syndicate", "bank", "participation", "roles"]
            },
            "V_INVESTOR_PORTFOLIO": {
                "domains": ["INVESTOR", "ORDER"],
                "type": "core",
                "priority": 6,
                "description": "Investor portfolio holdings and order history",
                "entities": ["investors", "portfolios", "orders"],
                "patterns": ["investor", "portfolio", "holdings", "history"]
            },
            "V_TRADE_EXECUTION": {
                "domains": ["TRADES", "ORDER", "INVESTOR"],
                "type": "core",
                "priority": 5,
                "description": "Trade execution details and order fulfillment",
                "entities": ["trades", "orders", "execution"],
                "patterns": ["trade", "execution", "fulfillment", "settlement"]
            },
            
            # Supporting Views
            "V_TRANCHE_INSTRUMENTS": {
                "domains": ["TRANCHE", "DEAL", "SYNDICATE"],
                "type": "supporting",
                "priority": 5,
                "description": "Detailed tranche instrument specifications",
                "entities": ["tranches", "instruments", "specifications"],
                "patterns": ["instrument", "specification", "details", "structure"]
            },
            "V_ORDER_DETAILS": {
                "domains": ["ORDER", "INVESTOR", "TRANCHE"],
                "type": "supporting",
                "priority": 4,
                "description": "Detailed order information and investor details",
                "entities": ["orders", "order_details", "investors"],
                "patterns": ["order", "details", "breakdown", "specifics"]
            },
            "V_ALLOCATION_SUMMARY": {
                "domains": ["ORDER", "SYNDICATE", "INVESTOR"],
                "type": "supporting",
                "priority": 4,
                "description": "Summary of allocation decisions by syndicate",
                "entities": ["allocations", "syndicate", "summary"],
                "patterns": ["allocation", "summary", "syndicate", "decisions"]
            },
            "V_TRADE_SETTLEMENT": {
                "domains": ["TRADES", "ORDER", "SYNDICATE"],
                "type": "supporting",
                "priority": 3,
                "description": "Trade settlement and post-trade processing",
                "entities": ["trades", "settlement", "processing"],
                "patterns": ["settlement", "processing", "post-trade", "clearing"]
            },
            "V_DEALER_TRADES": {
                "domains": ["TRADES", "ORDER"],
                "type": "supporting",
                "priority": 2,  # Lower priority for structural queries
                "description": "Dealer trading activity and execution details",
                "entities": ["dealers", "trades", "execution"],
                "patterns": ["dealer", "trading", "activity", "execution", "secondary"]
            }
        }
        
        success_count = 0
        for view_name, data in mappings_data.items():
            try:
                mapping = ViewDomainMapping(
                    view_name=view_name,
                    business_domains=data["domains"],
                    view_type=data["type"],
                    priority_score=data["priority"],
                    description=data["description"],
                    key_entities=data["entities"],
                    query_patterns=data["patterns"],
                    created_by="migration_script"
                )
                
                if self.metadata_service.add_view_mapping(mapping):
                    success_count += 1
                    logger.info(f"✅ Migrated {view_name}")
                else:
                    logger.error(f"❌ Failed to migrate {view_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error migrating {view_name}: {e}")
        
        logger.info(f"View domain mappings migration complete: {success_count}/{len(mappings_data)} successful")
    
    def migrate_view_dependencies(self):
        """Migrate hardcoded view dependencies to MongoDB."""
        logger.info("Starting view dependencies migration...")
        
        dependencies_data = {
            "V_TRANCHE_METRICS": {
                "supporting": ["V_TRANCHE_INSTRUMENTS", "V_TRANCHE_PRICING"],
                "type": "enhancement",
                "description": "Enhanced with detailed instrument specs and pricing data"
            },
            "V_ORDER_ALLOCATION": {
                "supporting": ["V_ORDER_DETAILS", "V_ALLOCATION_SUMMARY"],
                "type": "enhancement",
                "description": "Enhanced with detailed order information and allocation summaries"
            },
            "V_TRADE_EXECUTION": {
                "supporting": ["V_TRADE_SETTLEMENT", "V_ORDER_DETAILS"],
                "type": "enhancement",
                "description": "Enhanced with settlement details and order information"
            },
            "V_DEAL_SUMMARY": {
                "supporting": ["V_TRANCHE_METRICS", "V_SYNDICATE_PARTICIPATION"],
                "type": "enhancement",
                "description": "Enhanced with tranche metrics and syndicate participation data"
            }
        }
        
        success_count = 0
        for primary_view, data in dependencies_data.items():
            try:
                dependency = ViewDependency(
                    primary_view=primary_view,
                    supporting_views=data["supporting"],
                    dependency_type=data["type"],
                    description=data["description"]
                )
                
                # Insert into MongoDB
                result = self.metadata_service.view_dependencies_collection.replace_one(
                    {"primary_view": primary_view},
                    dependency.dict(),
                    upsert=True
                )
                
                success_count += 1
                logger.info(f"✅ Migrated dependency for {primary_view}")
                
            except Exception as e:
                logger.error(f"❌ Error migrating dependency for {primary_view}: {e}")
        
        logger.info(f"View dependencies migration complete: {success_count}/{len(dependencies_data)} successful")
    
    def migrate_query_patterns(self):
        """Migrate query patterns for better view selection."""
        logger.info("Starting query patterns migration...")
        
        # Enhanced patterns for V_TERMSHEET to ensure it ranks higher for deal/tranche queries
        patterns_data = {
            "V_TERMSHEET": {
                "keywords": ["deal", "tranche", "termsheet", "announced", "status", "fixed", "income", "names"],
                "patterns": [
                    r"deal\s+names?",
                    r"tranche.*announced",
                    r"announced.*status",
                    r"fixed\s+income",
                    r"deal.*tranche",
                    r"tranche.*deal"
                ],
                "context_indicators": [
                    "what are all the deal names",
                    "deal names with tranches",
                    "announced status",
                    "fixed income category",
                    "deal and tranche information"
                ],
                "samples": [
                    "What are all the deal names with tranches that are in announced status in Fixed Income?",
                    "Show me deals with announced tranches",
                    "List all fixed income deals with tranches"
                ],
                "keyword_weight": 2.0,
                "pattern_weight": 3.0,
                "context_weight": 4.0
            },
            "V_DEALER_TRADES": {
                "keywords": ["dealer", "trade", "trading", "execution", "secondary", "market"],
                "patterns": [
                    r"dealer.*trad",
                    r"trading.*activity",
                    r"secondary.*market",
                    r"trade.*execution"
                ],
                "context_indicators": [
                    "dealer trading activity",
                    "secondary market trades",
                    "trading execution details"
                ],
                "samples": [
                    "Show me dealer trading activity",
                    "What trades did dealers execute?"
                ],
                "keyword_weight": 1.0,
                "pattern_weight": 1.5,
                "context_weight": 1.0
            }
        }
        
        success_count = 0
        for view_name, data in patterns_data.items():
            try:
                pattern = ViewQueryPattern(
                    view_name=view_name,
                    query_keywords=data["keywords"],
                    query_patterns=data["patterns"],
                    business_context_indicators=data["context_indicators"],
                    sample_queries=data["samples"],
                    keyword_weight=data["keyword_weight"],
                    pattern_weight=data["pattern_weight"],
                    context_weight=data["context_weight"]
                )
                
                # Insert into MongoDB
                result = self.metadata_service.query_patterns_collection.replace_one(
                    {"view_name": view_name},
                    pattern.dict(),
                    upsert=True
                )
                
                success_count += 1
                logger.info(f"✅ Migrated query patterns for {view_name}")
                
            except Exception as e:
                logger.error(f"❌ Error migrating query patterns for {view_name}: {e}")
        
        logger.info(f"Query patterns migration complete: {success_count}/{len(patterns_data)} successful")
    
    def create_indexes(self):
        """Create MongoDB indexes for performance."""
        logger.info("Creating MongoDB indexes...")
        
        try:
            # View domain mappings indexes
            self.metadata_service.view_mappings_collection.create_index("view_name", unique=True)
            self.metadata_service.view_mappings_collection.create_index("business_domains")
            self.metadata_service.view_mappings_collection.create_index("view_type")
            self.metadata_service.view_mappings_collection.create_index("priority_score")
            
            # View dependencies indexes
            self.metadata_service.view_dependencies_collection.create_index("primary_view", unique=True)
            self.metadata_service.view_dependencies_collection.create_index("supporting_views")
            
            # Query patterns indexes
            self.metadata_service.query_patterns_collection.create_index("view_name", unique=True)
            self.metadata_service.query_patterns_collection.create_index("query_keywords")
            
            # Usage stats indexes
            self.metadata_service.usage_stats_collection.create_index("view_name", unique=True)
            self.metadata_service.usage_stats_collection.create_index("last_used")
            self.metadata_service.usage_stats_collection.create_index("usage_count")
            
            logger.info("✅ Created MongoDB indexes successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
    
    def run_full_migration(self):
        """Run complete migration process."""
        logger.info("🚀 Starting full view metadata migration...")
        
        try:
            # Step 1: Migrate domain mappings
            self.migrate_view_domain_mappings()
            
            # Step 2: Migrate dependencies
            self.migrate_view_dependencies()
            
            # Step 3: Migrate query patterns
            self.migrate_query_patterns()
            
            # Step 4: Create indexes
            self.create_indexes()
            
            # Step 5: Verify migration
            self.verify_migration()
            
            logger.info("🎉 View metadata migration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        
        finally:
            self.metadata_service.close()
    
    def verify_migration(self):
        """Verify the migration was successful."""
        logger.info("Verifying migration...")
        
        # Check mappings
        mappings = self.metadata_service.get_view_domain_mappings()
        logger.info(f"✅ Verified {len(mappings)} view domain mappings")
        
        # Check dependencies
        dependencies = self.metadata_service.get_view_dependencies()
        logger.info(f"✅ Verified {len(dependencies)} view dependencies")
        
        # Check patterns
        patterns = self.metadata_service.get_query_patterns()
        logger.info(f"✅ Verified {len(patterns)} query patterns")
        
        # Test specific mappings
        if "V_TERMSHEET" in mappings:
            termsheet_domains = mappings["V_TERMSHEET"]
            if "DEAL" in termsheet_domains and "TRANCHE" in termsheet_domains:
                logger.info("✅ V_TERMSHEET correctly mapped to DEAL and TRANCHE domains")
            else:
                logger.warning("⚠️  V_TERMSHEET domain mapping may be incorrect")
        
        logger.info("Migration verification complete")


def main():
    """Main migration function."""
    print("🏗️  View Metadata Migration Script")
    print("=" * 50)
    print("This script will migrate hardcoded view metadata to MongoDB.")
    print("It will create view domain mappings, dependencies, and query patterns.")
    print()
    
    # Confirm before proceeding
    response = input("Do you want to proceed with the migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    try:
        migrator = ViewMetadataMigrator()
        migrator.run_full_migration()
        
        print("\n🎉 Migration Summary:")
        print("  ✅ View domain mappings migrated")
        print("  ✅ View dependencies migrated")
        print("  ✅ Query patterns migrated")
        print("  ✅ MongoDB indexes created")
        print("\nNext steps:")
        print("  1. Restart your application to use MongoDB-based view metadata")
        print("  2. Monitor logs to ensure ViewSelectionService loads from MongoDB")
        print("  3. Test with the query: 'What are all the deal names with tranches in announced status?'")
        print("  4. V_TERMSHEET should now be prioritized over V_DEALER_TRADES")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Check the logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()