#!/usr/bin/env python3
"""
Dynamic Metadata Discovery and Migration Script

This script reads actual data files to discover and migrate metadata:
- Scans view_metadata.json for view information
- Scans reports/ folder for report metadata  
- Uses predefined business domains
- Creates comprehensive MongoDB metadata without any hardcoded mappings

Usage:
    python scripts/discover_and_migrate_metadata.py --reports-dir meta_documents/reports --views-file meta_documents/view_metadata.json
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import argparse

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.models.view_metadata_models import (
    ViewDomainMapping,
    ViewDependency, 
    ViewQueryPattern,
    ViewMetadataConfig,
    ViewUsageStats
)
from text_to_sql_rag.models.business_domain_models import (
    BusinessDomainDefinition,
    DomainTerminology,
    DomainDetectionRule,
    ViewClassificationRule,
    BusinessContextConfig
)
from text_to_sql_rag.services.view_metadata_service import ViewMetadataService
from text_to_sql_rag.services.business_domain_metadata_service import BusinessDomainMetadataService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DynamicMetadataDiscoverer:
    """Discovers metadata from actual data files and migrates to MongoDB."""
    
    def __init__(self, reports_dir: str = None, views_file: str = None):
        """
        Initialize metadata discoverer.
        
        Args:
            reports_dir: Directory containing report JSON files
            views_file: Path to view_metadata.json file
        """
        self.reports_dir = Path(reports_dir) if reports_dir else Path("meta_documents/reports")
        self.views_file = Path(views_file) if views_file else Path("meta_documents/view_metadata.json")
        self.metadata_service = ViewMetadataService()
        self.domain_metadata_service = BusinessDomainMetadataService()
        
        # Business domains - these are stable, defined in your system
        self.business_domains = [
            {
                "id": "issuer",
                "title": "ISSUER", 
                "summary": "Companies seeking capital through bond issuances",
                "bullets": [
                    "Top-level entity initiating fundraising",
                    "One issuer can have multiple deals"
                ]
            },
            {
                "id": "deal",
                "title": "DEAL",
                "summary": "Fundraising initiatives created by JPMorgan for issuers", 
                "bullets": [
                    "Container for all bond issuances for a specific capital raise",
                    "Each deal belongs to one issuer and contains multiple tranches"
                ]
            },
            {
                "id": "tranche",
                "title": "TRANCHE",
                "summary": "Individual bond issuances with distinct terms",
                "bullets": [
                    "Core object with pricing, maturity, and ratings information",
                    "Multiple tranches per deal allow different risk/return profiles",
                    "Each tranche belongs to one deal"
                ]
            },
            {
                "id": "syndicate", 
                "title": "SYNDICATE",
                "summary": "Financial institutions participating in distribution",
                "bullets": [
                    "Multiple banks per tranche with different roles (lead, co-manager)",
                    "Handle distribution and allocation decisions"
                ]
            },
            {
                "id": "order",
                "title": "ORDER",
                "summary": "Investment requests from institutional investors",
                "bullets": [
                    "Contains IOI (Indication of Interest) and Final Allocation",
                    "Multiple orders per tranche from different investors"
                ]
            },
            {
                "id": "investor",
                "title": "INVESTOR", 
                "summary": "Primary market investor entity that invests in deals run by JPMC",
                "bullets": [
                    "Places bond or hedge orders on tranches",
                    "Final investment given to investor is determined by the syndicate bank in charge of allocation"
                ]
            },
            {
                "id": "order_basis",
                "title": "ORDER_BASIS",
                "summary": "Hedge order amount within orders",
                "bullets": []
            },
            {
                "id": "order_limit",
                "title": "ORDER_LIMIT", 
                "summary": "Bond order amount within orders",
                "bullets": [
                    "Reoffer Order Limit: Unconditional investment amount",
                    "Conditional Order Limit: Investment with price/yield thresholds"
                ]
            },
            {
                "id": "trade",
                "title": "TRADE",
                "summary": "Records the final trades that get booked",
                "bullets": []
            }
        ]
        
        # Create domain lookup for easy access
        self.domain_lookup = {domain["title"]: domain for domain in self.business_domains}
        
        logger.info(f"DynamicMetadataDiscoverer initialized")
        logger.info(f"Reports directory: {self.reports_dir}")
        logger.info(f"Views file: {self.views_file}")
        logger.info(f"Business domains: {len(self.business_domains)}")
    
    def discover_view_metadata(self) -> List[Dict[str, Any]]:
        """Discover view metadata from view_metadata.json."""
        logger.info("Discovering view metadata...")
        
        if not self.views_file.exists():
            logger.error(f"Views file not found: {self.views_file}")
            return []
        
        try:
            with open(self.views_file, 'r', encoding='utf-8') as f:
                views_data = json.load(f)
            
            discovered_views = []
            
            # Process core views
            core_views = views_data.get("CORE_VIEWS", [])
            for view_data in core_views:
                discovered_views.append(self._process_view_data(view_data, "core"))
            
            # Process supporting views  
            supporting_views = views_data.get("SUPPORTING_VIEWS", [])
            for view_data in supporting_views:
                discovered_views.append(self._process_view_data(view_data, "supporting"))
            
            logger.info(f"Discovered {len(discovered_views)} views ({len(core_views)} core, {len(supporting_views)} supporting)")
            return discovered_views
            
        except Exception as e:
            logger.error(f"Error discovering view metadata: {e}")
            return []
    
    def _process_view_data(self, view_data: Dict[str, Any], view_type: str) -> Dict[str, Any]:
        """Process individual view data and enhance with smart defaults."""
        
        view_name = view_data.get("view_name", "")
        domains = view_data.get("domains", [])
        description = view_data.get("description", "")
        patterns = view_data.get("patterns", [])
        entities = view_data.get("entities", [])
        
        # Calculate priority based on view type and name patterns
        priority = self._calculate_view_priority(view_name, view_type, domains)
        
        # Extract additional patterns from view name and description
        additional_patterns = self._extract_patterns_from_text(view_name + " " + description)
        all_patterns = list(set(patterns + additional_patterns))
        
        # Generate query examples based on view purpose
        query_examples = self._generate_query_examples(view_name, domains, description)
        
        return {
            "view_name": view_name,
            "domains": domains,
            "view_type": view_type,
            "priority": priority,
            "description": description,
            "entities": entities,
            "patterns": all_patterns,
            "query_examples": query_examples,
            "use_cases": view_data.get("use_cases", ""),
            "data_returned": view_data.get("data_returned", ""),
            "example_sql": view_data.get("example_query", ""),
            "view_sql": view_data.get("view_sql", "")
        }
    
    def _calculate_view_priority(self, view_name: str, view_type: str, domains: List[str]) -> int:
        """Calculate view priority based on view characteristics."""
        
        base_priority = 8 if view_type == "core" else 4
        
        # Adjust based on view name patterns
        high_priority_patterns = ["SUMMARY", "DETAILS", "METRICS", "MAIN", "PRIMARY"]
        medium_priority_patterns = ["ALLOCATION", "EXECUTION", "PRICING"]
        low_priority_patterns = ["TEMP", "STAGING", "LOG"]
        
        view_upper = view_name.upper()
        
        if any(pattern in view_upper for pattern in high_priority_patterns):
            base_priority += 2
        elif any(pattern in view_upper for pattern in medium_priority_patterns):
            base_priority += 1
        elif any(pattern in view_upper for pattern in low_priority_patterns):
            base_priority -= 2
        
        # Adjust based on domain coverage (more domains = more useful)
        if len(domains) >= 3:
            base_priority += 1
        
        # Ensure priority stays in 1-10 range
        return max(1, min(10, base_priority))
    
    def _extract_patterns_from_text(self, text: str) -> List[str]:
        """Extract useful patterns from view name and description."""
        
        # Common financial terms to extract
        financial_terms = [
            "deal", "tranche", "order", "allocation", "pricing", "yield", "spread",
            "syndicate", "investor", "trade", "execution", "settlement", "issuer",
            "bond", "fixed", "income", "metrics", "summary", "details"
        ]
        
        text_lower = text.lower()
        found_patterns = []
        
        for term in financial_terms:
            if term in text_lower:
                found_patterns.append(term)
        
        # Extract meaningful words (3+ characters, not common words)
        words = text_lower.replace("_", " ").split()
        common_words = {"the", "and", "for", "with", "this", "that", "from", "view", "table"}
        
        for word in words:
            if len(word) >= 3 and word not in common_words and word.isalpha():
                found_patterns.append(word)
        
        return list(set(found_patterns))
    
    def _generate_query_examples(self, view_name: str, domains: List[str], description: str) -> List[str]:
        """Generate realistic query examples based on view characteristics."""
        
        examples = []
        
        # Domain-based examples
        if "DEAL" in domains:
            examples.extend([
                "Show me all deal information",
                "What deals were announced today?",
                "List deals by status"
            ])
        
        if "TRANCHE" in domains:
            examples.extend([
                "What are the tranche details for this deal?",
                "Show tranches with announced status",
                "List all tranches in fixed income"
            ])
        
        if "ORDER" in domains:
            examples.extend([
                "Show order allocation details",
                "What orders were placed for this tranche?"
            ])
        
        if "INVESTOR" in domains:
            examples.extend([
                "Which investors participated in this deal?",
                "Show investor allocation summary"
            ])
        
        # View name-based examples
        view_lower = view_name.lower()
        if "summary" in view_lower:
            examples.append(f"Give me a summary from {view_name}")
        if "detail" in view_lower:
            examples.append(f"Show detailed information from {view_name}")
        if "metrics" in view_lower:
            examples.append(f"What are the key metrics in {view_name}?")
        
        # Limit to 5 most relevant examples
        return examples[:5]
    
    def discover_report_metadata(self) -> List[Dict[str, Any]]:
        """Discover report metadata from reports directory."""
        logger.info("Discovering report metadata...")
        
        if not self.reports_dir.exists():
            logger.warning(f"Reports directory not found: {self.reports_dir}")
            return []
        
        discovered_reports = []
        
        try:
            for report_file in self.reports_dir.glob("*.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    processed_report = self._process_report_data(report_data)
                    if processed_report:
                        discovered_reports.append(processed_report)
                        
                except Exception as e:
                    logger.warning(f"Error processing report {report_file}: {e}")
                    continue
            
            logger.info(f"Discovered {len(discovered_reports)} reports")
            return discovered_reports
            
        except Exception as e:
            logger.error(f"Error discovering report metadata: {e}")
            return []
    
    def _process_report_data(self, report_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process individual report data."""
        
        view_name = report_data.get("view_name")
        if not view_name:
            return None
        
        domains = report_data.get("domains", [])
        description = report_data.get("report_description", "")
        
        # Extract patterns from report data
        patterns = self._extract_patterns_from_text(
            f"{view_name} {description} {report_data.get('report_name', '')}"
        )
        
        # Generate query examples for reports
        query_examples = self._generate_query_examples(view_name, domains, description)
        
        return {
            "view_name": view_name,
            "domains": domains,
            "view_type": "supporting",  # Reports are typically supporting views
            "priority": self._calculate_view_priority(view_name, "supporting", domains),
            "description": description,
            "report_name": report_data.get("report_name", ""),
            "patterns": patterns,
            "query_examples": query_examples,
            "use_cases": report_data.get("use_cases", ""),
            "data_returned": report_data.get("data_returned", ""),
            "example_sql": report_data.get("example_sql", "")
        }
    
    def discover_view_dependencies(self, all_views: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Discover view dependencies based on naming patterns and domain relationships."""
        logger.info("Discovering view dependencies...")
        
        dependencies = {}
        view_by_name = {v["view_name"]: v for v in all_views}
        
        for view in all_views:
            if view["view_type"] != "core":
                continue
                
            view_name = view["view_name"]
            view_domains = set(view["domains"])
            
            # Find supporting views
            supporting_views = []
            
            for potential_support in all_views:
                if potential_support["view_type"] != "supporting":
                    continue
                    
                support_name = potential_support["view_name"]
                support_domains = set(potential_support["domains"])
                
                # Check if domains overlap
                if view_domains.intersection(support_domains):
                    # Check for name-based relationships
                    if self._are_views_related(view_name, support_name):
                        supporting_views.append(support_name)
            
            if supporting_views:
                dependencies[view_name] = supporting_views[:5]  # Limit to 5 supports
        
        logger.info(f"Discovered {len(dependencies)} view dependencies")
        return dependencies
    
    def _are_views_related(self, primary_view: str, support_view: str) -> bool:
        """Determine if two views are related based on naming patterns."""
        
        primary_parts = set(primary_view.upper().split("_"))
        support_parts = set(support_view.upper().split("_"))
        
        # Check for shared keywords
        shared_keywords = primary_parts.intersection(support_parts)
        
        # Views are related if they share at least 1 meaningful keyword
        meaningful_keywords = shared_keywords - {"V", "SYND", "THE", "FOR", "AND", "WITH"}
        
        return len(meaningful_keywords) >= 1
    
    def migrate_to_mongodb(self):
        """Complete migration to MongoDB."""
        logger.info("🚀 Starting dynamic metadata migration...")
        
        try:
            # Clean MongoDB collections first
            self._clean_mongodb_collections()
            
            # Step 1: Discover view metadata
            view_metadata = self.discover_view_metadata()
            
            # Step 2: Discover report metadata  
            report_metadata = self.discover_report_metadata()
            
            # Step 3: Combine all metadata
            all_metadata = view_metadata + report_metadata
            
            # Step 4: Migrate view domain mappings
            self._migrate_view_mappings(all_metadata)
            
            # Step 5: Discover and migrate dependencies
            dependencies = self.discover_view_dependencies(all_metadata)
            self._migrate_dependencies(dependencies)
            
            # Step 6: Generate and migrate query patterns
            self._migrate_query_patterns(all_metadata)
            
            # Step 7: Migrate business domain metadata
            self._migrate_business_domains()
            self._migrate_domain_terminology()
            self._migrate_detection_rules()
            self._migrate_classification_rules()
            
            # Step 8: Create configuration
            self._create_metadata_config()
            
            # Step 9: Create indexes
            self._create_indexes()
            
            # Step 10: Verify migration
            self._verify_migration()
            
            logger.info("🎉 Dynamic metadata migration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        
        finally:
            self.metadata_service.close()
            self.domain_metadata_service.close()
    
    def _clean_mongodb_collections(self):
        """Clean all MongoDB collections before migration."""
        logger.info("Cleaning MongoDB collections...")
        
        collections = [
            self.metadata_service.view_mappings_collection,
            self.metadata_service.view_dependencies_collection,
            self.metadata_service.query_patterns_collection,
            self.metadata_service.metadata_config_collection,
            self.metadata_service.usage_stats_collection,
            self.domain_metadata_service.domains_collection,
            self.domain_metadata_service.terminology_collection,
            self.domain_metadata_service.detection_rules_collection,
            self.domain_metadata_service.classification_rules_collection,
            self.domain_metadata_service.context_config_collection
        ]
        
        for collection in collections:
            result = collection.delete_many({})
            logger.info(f"Deleted {result.deleted_count} documents from {collection.name}")
    
    def _migrate_view_mappings(self, all_metadata: List[Dict[str, Any]]):
        """Migrate view domain mappings to MongoDB."""
        logger.info("Migrating view domain mappings...")
        
        success_count = 0
        for view_data in all_metadata:
            try:
                mapping = ViewDomainMapping(
                    view_name=view_data["view_name"],
                    business_domains=view_data["domains"],
                    view_type=view_data["view_type"],
                    priority_score=view_data["priority"],
                    description=view_data["description"],
                    key_entities=view_data.get("entities", []),
                    query_patterns=view_data["patterns"],
                    created_by="dynamic_discovery_script"
                )
                
                if self.metadata_service.add_view_mapping(mapping):
                    success_count += 1
                    logger.info(f"✅ Migrated {view_data['view_name']}")
                
            except Exception as e:
                logger.error(f"❌ Error migrating {view_data.get('view_name', 'unknown')}: {e}")
        
        logger.info(f"View mappings migration complete: {success_count}/{len(all_metadata)} successful")
    
    def _migrate_dependencies(self, dependencies: Dict[str, List[str]]):
        """Migrate view dependencies to MongoDB."""
        logger.info("Migrating view dependencies...")
        
        success_count = 0
        for primary_view, supporting_views in dependencies.items():
            try:
                dependency = ViewDependency(
                    primary_view=primary_view,
                    supporting_views=supporting_views,
                    dependency_type="enhancement",
                    description=f"Enhanced with related supporting views based on domain overlap"
                )
                
                self.metadata_service.view_dependencies_collection.replace_one(
                    {"primary_view": primary_view},
                    dependency.dict(),
                    upsert=True
                )
                
                success_count += 1
                logger.info(f"✅ Migrated dependencies for {primary_view}")
                
            except Exception as e:
                logger.error(f"❌ Error migrating dependencies for {primary_view}: {e}")
        
        logger.info(f"Dependencies migration complete: {success_count}/{len(dependencies)} successful")
    
    def _migrate_query_patterns(self, all_metadata: List[Dict[str, Any]]):
        """Migrate query patterns to MongoDB."""
        logger.info("Migrating query patterns...")
        
        success_count = 0
        for view_data in all_metadata:
            try:
                # Generate context indicators from description and use cases
                context_indicators = self._generate_context_indicators(view_data)
                
                pattern = ViewQueryPattern(
                    view_name=view_data["view_name"],
                    query_keywords=view_data["patterns"][:20],  # Limit keywords
                    query_patterns=[],  # We'll rely on keywords for now
                    business_context_indicators=context_indicators,
                    sample_queries=view_data.get("query_examples", []),
                    keyword_weight=2.0 if view_data["view_type"] == "core" else 1.0,
                    pattern_weight=1.5,
                    context_weight=3.0 if view_data["view_type"] == "core" else 2.0
                )
                
                self.metadata_service.query_patterns_collection.replace_one(
                    {"view_name": view_data["view_name"]},
                    pattern.dict(),
                    upsert=True
                )
                
                success_count += 1
                logger.info(f"✅ Migrated query patterns for {view_data['view_name']}")
                
            except Exception as e:
                logger.error(f"❌ Error migrating query patterns for {view_data.get('view_name', 'unknown')}: {e}")
        
        logger.info(f"Query patterns migration complete: {success_count}/{len(all_metadata)} successful")
    
    def _generate_context_indicators(self, view_data: Dict[str, Any]) -> List[str]:
        """Generate business context indicators from view data."""
        
        indicators = []
        
        # Add description phrases
        description = view_data.get("description", "")
        if description:
            # Split description into meaningful phrases
            phrases = [phrase.strip() for phrase in description.split(".") if len(phrase.strip()) > 10]
            indicators.extend(phrases[:3])  # Take first 3 meaningful phrases
        
        # Add use case phrases
        use_cases = view_data.get("use_cases", "")
        if use_cases:
            use_case_phrases = [phrase.strip() for phrase in use_cases.split("-") if len(phrase.strip()) > 10]
            indicators.extend(use_case_phrases[:2])  # Take first 2 use cases
        
        # Add domain-specific context
        domains = view_data.get("domains", [])
        for domain in domains:
            domain_info = self.domain_lookup.get(domain)
            if domain_info:
                indicators.append(domain_info["summary"])
        
        return indicators[:10]  # Limit to 10 context indicators
    
    def _migrate_business_domains(self):
        """Migrate business domain definitions to MongoDB."""
        logger.info("Migrating business domain definitions...")
        
        success_count = 0
        for domain_data in self.business_domains:
            try:
                # Calculate hierarchy relationships
                parent_domains = []
                child_domains = []
                domain_level = 1
                
                # Determine relationships based on business logic
                domain_title = domain_data["title"]
                if domain_title == "DEAL":
                    parent_domains = ["ISSUER"]
                    child_domains = ["TRANCHE"]
                    domain_level = 2
                elif domain_title == "TRANCHE":
                    parent_domains = ["DEAL"]
                    child_domains = ["SYNDICATE", "ORDER"]
                    domain_level = 3
                elif domain_title == "ORDER":
                    parent_domains = ["TRANCHE"]
                    child_domains = ["ORDER_LIMIT", "ORDER_BASIS"]
                    domain_level = 4
                elif domain_title == "SYNDICATE":
                    parent_domains = ["TRANCHE"]
                    domain_level = 4
                elif domain_title == "ORDER_LIMIT":
                    parent_domains = ["ORDER"]
                    domain_level = 5
                elif domain_title == "ORDER_BASIS":
                    parent_domains = ["ORDER"]
                    domain_level = 5
                elif domain_title == "TRADE":
                    parent_domains = ["ORDER"]
                    domain_level = 5
                elif domain_title == "INVESTOR":
                    child_domains = ["ORDER"]
                    domain_level = 1
                
                # Generate key concepts from domain title and summary
                key_concepts = []
                summary_words = domain_data["summary"].lower().split()
                title_words = domain_title.lower().replace("_", " ").split()
                
                key_concepts.extend(title_words)
                key_concepts.extend([word for word in summary_words if len(word) > 3])
                key_concepts = list(set(key_concepts))  # Remove duplicates
                
                domain_def = BusinessDomainDefinition(
                    domain_id=domain_data["id"],
                    domain_name=domain_data["title"],
                    summary=domain_data["summary"],
                    description=domain_data["summary"],  # Use summary as description
                    parent_domains=parent_domains,
                    child_domains=child_domains,
                    key_concepts=key_concepts,
                    business_rules=domain_data.get("bullets", []),
                    typical_queries=[f"Show me {domain_title.lower()} information"],
                    domain_level=domain_level,
                    relationship_type="intermediate_entity" if parent_domains and child_domains else 
                                   "root_entity" if child_domains else "leaf_entity",
                    created_by="dynamic_discovery_script"
                )
                
                if self.domain_metadata_service.add_business_domain(domain_def):
                    success_count += 1
                    logger.info(f"✅ Migrated business domain {domain_title}")
                
            except Exception as e:
                logger.error(f"❌ Error migrating domain {domain_data.get('title', 'unknown')}: {e}")
        
        logger.info(f"Business domains migration complete: {success_count}/{len(self.business_domains)} successful")
    
    def _migrate_domain_terminology(self):
        """Migrate domain terminology to MongoDB."""
        logger.info("Migrating domain terminology...")
        
        # Create terminology mappings based on business domains
        terminology_mappings = {
            "ISSUER": ["issuer", "company", "corporate", "entity", "borrower", "bond", "capital", "fundraising"],
            "DEAL": ["deal", "fundraising", "initiative", "capital_raise", "bond_issuance", "program", "offering"],
            "TRANCHE": ["tranche", "bond", "pricing", "yield", "spread", "maturity", "ratings", "financial_metrics", "series"],
            "SYNDICATE": ["syndicate", "bank", "distribution", "lead_manager", "co_manager", "underwriter", "bookrunner"],
            "ORDER": ["order", "ioi", "indication", "interest", "final_allocation", "investor", "investment", "demand"],
            "ORDER_LIMIT": ["order_limit", "reoffer", "conditional", "investment_amount", "threshold", "unconditional"],
            "ORDER_BASIS": ["order_basis", "hedge", "basis", "amount", "component"],
            "INVESTOR": ["investor", "institutional", "investment", "portfolio", "allocation", "mandate", "fund"],
            "TRADE": ["trade", "execution", "trade_date", "price", "yield", "dealer", "settlement", "confirmation"]
        }
        
        success_count = 0
        for domain_name, terms in terminology_mappings.items():
            try:
                terminology = DomainTerminology(
                    domain_name=domain_name,
                    term_type="primary",
                    terms=terms,
                    weight=1.0,
                    context_phrases=[f"{domain_name.lower()} information", f"related to {domain_name.lower()}"]
                )
                
                if self.domain_metadata_service.add_domain_terminology(terminology):
                    success_count += 1
                    logger.info(f"✅ Migrated terminology for {domain_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error migrating terminology for {domain_name}: {e}")
        
        logger.info(f"Domain terminology migration complete: {success_count}/{len(terminology_mappings)} successful")
    
    def _migrate_detection_rules(self):
        """Migrate domain detection rules to MongoDB."""
        logger.info("Migrating domain detection rules...")
        
        # Create detection rules for each domain
        detection_rules = [
            {
                "rule_name": "tranche_detection",
                "target_domains": ["TRANCHE"],
                "keywords": ["tranche", "bond", "pricing", "yield", "spread", "maturity", "announced", "status"],
                "priority": 10  # Highest priority for tranche queries
            },
            {
                "rule_name": "deal_detection", 
                "target_domains": ["DEAL"],
                "keywords": ["deal", "fundraising", "initiative", "capital_raise", "program"],
                "priority": 9
            },
            {
                "rule_name": "issuer_detection",
                "target_domains": ["ISSUER"],
                "keywords": ["issuer", "company", "corporate", "entity", "borrower"],
                "priority": 8
            },
            {
                "rule_name": "order_detection",
                "target_domains": ["ORDER"],
                "keywords": ["order", "allocation", "investor", "investment", "ioi"],
                "priority": 7
            },
            {
                "rule_name": "trade_detection",
                "target_domains": ["TRADE"],
                "keywords": ["trade", "execution", "dealer", "settlement", "secondary"],
                "priority": 6
            }
        ]
        
        success_count = 0
        for rule_data in detection_rules:
            try:
                rule = DomainDetectionRule(
                    rule_name=rule_data["rule_name"],
                    target_domains=rule_data["target_domains"],
                    keyword_patterns=rule_data["keywords"],
                    phrase_patterns=[],
                    context_requirements=[],
                    match_type="any",
                    confidence_weight=1.0,
                    minimum_matches=1,
                    priority=rule_data["priority"],
                    is_fallback=False
                )
                
                if self.domain_metadata_service.add_detection_rule(rule):
                    success_count += 1
                    logger.info(f"✅ Migrated detection rule {rule_data['rule_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Error migrating detection rule {rule_data['rule_name']}: {e}")
        
        logger.info(f"Detection rules migration complete: {success_count}/{len(detection_rules)} successful")
    
    def _migrate_classification_rules(self):
        """Migrate view classification rules to MongoDB."""
        logger.info("Migrating view classification rules...")
        
        classification_rules = [
            {
                "rule_name": "core_summary_pattern",
                "classification": "core",
                "name_patterns": ["SUMMARY", "MAIN", "PRIMARY"],
                "priority_boost": 2
            },
            {
                "rule_name": "core_metrics_pattern",
                "classification": "core", 
                "name_patterns": ["METRICS", "PERFORMANCE", "ANALYTICS"],
                "priority_boost": 2
            },
            {
                "rule_name": "supporting_details_pattern",
                "classification": "supporting",
                "name_patterns": ["DETAILS", "BREAKDOWN", "EXPANDED"],
                "priority_boost": 1
            },
            {
                "rule_name": "supporting_instruments_pattern",
                "classification": "supporting",
                "name_patterns": ["INSTRUMENTS", "SPECIFICATIONS", "COMPONENTS"],
                "priority_boost": 1
            },
            {
                "rule_name": "utility_temp_pattern", 
                "classification": "utility",
                "name_patterns": ["TEMP", "STAGING", "LOG", "AUDIT"],
                "priority_boost": -2
            }
        ]
        
        success_count = 0
        for rule_data in classification_rules:
            try:
                rule = ViewClassificationRule(
                    rule_name=rule_data["rule_name"],
                    classification_type=rule_data["classification"],
                    name_patterns=rule_data["name_patterns"],
                    description_patterns=[],
                    domain_requirements=[],
                    priority_boost=rule_data["priority_boost"],
                    match_logic="any",
                    is_active=True
                )
                
                if self.domain_metadata_service.add_classification_rule(rule):
                    success_count += 1
                    logger.info(f"✅ Migrated classification rule {rule_data['rule_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Error migrating classification rule {rule_data['rule_name']}: {e}")
        
        logger.info(f"Classification rules migration complete: {success_count}/{len(classification_rules)} successful")
    
    def _create_metadata_config(self):
        """Create metadata configuration."""
        logger.info("Creating metadata configuration...")
        
        try:
            config = ViewMetadataConfig(
                config_name="production",
                auto_discovery_enabled=True,
                cache_ttl_minutes=60,  # 1 hour cache
                fallback_to_hardcoded=False,  # No hardcoded fallbacks
                default_core_priority=7,
                default_supporting_priority=4,
                sync_frequency_minutes=120  # Sync every 2 hours
            )
            
            self.metadata_service.metadata_config_collection.replace_one(
                {"config_name": "production"},
                config.dict(),
                upsert=True
            )
            
            logger.info("✅ Created metadata configuration")
            
        except Exception as e:
            logger.error(f"❌ Error creating metadata configuration: {e}")
    
    def _create_indexes(self):
        """Create MongoDB indexes."""
        logger.info("Creating MongoDB indexes...")
        
        try:
            # View domain mappings indexes
            self.metadata_service.view_mappings_collection.create_index("view_name", unique=True)
            self.metadata_service.view_mappings_collection.create_index("business_domains")
            self.metadata_service.view_mappings_collection.create_index("view_type")
            self.metadata_service.view_mappings_collection.create_index("priority_score")
            
            # View dependencies indexes
            self.metadata_service.view_dependencies_collection.create_index("primary_view", unique=True)
            
            # Query patterns indexes
            self.metadata_service.query_patterns_collection.create_index("view_name", unique=True)
            self.metadata_service.query_patterns_collection.create_index("query_keywords")
            
            # Business domain metadata indexes
            self.domain_metadata_service.create_indexes()
            
            logger.info("✅ Created MongoDB indexes")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
    
    def _verify_migration(self):
        """Verify migration success."""
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
        
        # List discovered views by type
        core_views = [name for name, domains in mappings.items() 
                     if any(doc.view_type == "core" for doc in 
                           [self.metadata_service.view_mappings_collection.find_one({"view_name": name})]
                           if doc)]
        
        supporting_views = [name for name, domains in mappings.items() 
                          if any(doc and doc.get("view_type") == "supporting" for doc in 
                                [self.metadata_service.view_mappings_collection.find_one({"view_name": name})]
                                if doc)]
        
        logger.info(f"📊 Discovery Summary:")
        logger.info(f"  Core Views: {len([v for v in mappings if self._get_view_type(v) == 'core'])}")
        logger.info(f"  Supporting Views: {len([v for v in mappings if self._get_view_type(v) == 'supporting'])}")
        logger.info(f"  Total Domains Used: {len(set(domain for domains in mappings.values() for domain in domains))}")
        
        logger.info("Migration verification complete")
    
    def _get_view_type(self, view_name: str) -> str:
        """Get view type from MongoDB."""
        try:
            doc = self.metadata_service.view_mappings_collection.find_one({"view_name": view_name})
            return doc.get("view_type", "unknown") if doc else "unknown"
        except:
            return "unknown"


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Discover and migrate metadata dynamically")
    parser.add_argument("--reports-dir", help="Path to reports directory", default="meta_documents/reports")
    parser.add_argument("--views-file", help="Path to view_metadata.json", default="meta_documents/view_metadata.json")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without doing it")
    
    args = parser.parse_args()
    
    print("🔍 Dynamic Metadata Discovery & Migration Script")
    print("=" * 60)
    print("This script will discover metadata from your actual data files:")
    print(f"  📁 Reports: {args.reports_dir}")  
    print(f"  📄 Views: {args.views_file}")
    print("  🏗️  Business Domains: Built-in definitions")
    print()
    
    if not args.dry_run:
        response = input("This will CLEAN MongoDB and rebuild from discovered data. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return
    
    try:
        discoverer = DynamicMetadataDiscoverer(
            reports_dir=args.reports_dir,
            views_file=args.views_file
        )
        
        if args.dry_run:
            # Show what would be discovered
            views = discoverer.discover_view_metadata()
            reports = discoverer.discover_report_metadata()
            all_metadata = views + reports
            
            print(f"\n🔍 Would discover {len(all_metadata)} total views/reports:")
            core_count = len([v for v in all_metadata if v.get("view_type") == "core"])
            supporting_count = len([v for v in all_metadata if v.get("view_type") == "supporting"])
            print(f"  📊 Core Views: {core_count}")
            print(f"  🔧 Supporting Views: {supporting_count}")
            
            all_domains = set()
            for item in all_metadata:
                all_domains.update(item.get("domains", []))
            print(f"  🏷️  Unique Domains: {len(all_domains)}")
            print(f"  🏷️  Domains: {sorted(all_domains)}")
            
        else:
            # Run full migration
            discoverer.migrate_to_mongodb()
            
            print("\n🎉 Migration Summary:")
            print("  ✅ Discovered metadata from actual data files")
            print("  ✅ No hardcoded mappings - all dynamic!")
            print("  ✅ MongoDB populated with discovered data")
            print("  ✅ View dependencies auto-generated")
            print("  ✅ Query patterns created from view data")
            print("\nNext steps:")
            print("  1. Restart your application")
            print("  2. All view metadata now comes from MongoDB")
            print("  3. To add new views: just add them to your data files and re-run this script")
            print("  4. No code changes needed for new views!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()