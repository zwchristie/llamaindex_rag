#!/usr/bin/env python3
"""
Business Domain Processing Script for Business Domain-First Architecture.
Creates structured business domain hierarchy definitions based on user's financial instruments knowledge.
Generates entity relationship definitions and business rules for RAG consumption.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BusinessDomainProcessor:
    """Process business domain definitions into structured RAG-ready content."""
    
    def __init__(self, output_dir: str):
        """
        Initialize business domain processor.
        
        Args:
            output_dir: Directory to save processed business domain files
        """
        self.output_dir = Path(output_dir)
        self.business_domains_dir = self.output_dir / "business_domains"
        
        # Create directories
        self.business_domains_dir.mkdir(parents=True, exist_ok=True)
        
        # User's business domain hierarchy from their specifications
        self.business_hierarchy = {
            "ISSUER": {
                "description": "Companies seeking capital through bond issuances",
                "details": [
                    "Top-level entity initiating fundraising",
                    "One issuer can have multiple deals",
                    "Primary market participant seeking capital"
                ],
                "parent_domains": [],
                "child_domains": ["DEAL"],
                "key_concepts": [
                    "issuer", "company", "capital", "bond", "issuance", 
                    "fundraising", "corporate", "entity", "borrower"
                ],
                "business_rules": [
                    "Each issuer can create multiple deals",
                    "Issuer information must be complete before deal creation",
                    "Credit rating affects deal terms and pricing"
                ],
                "typical_queries": [
                    "issuer information and credit ratings",
                    "deals by issuer",
                    "issuer fundraising history"
                ]
            },
            
            "DEAL": {
                "description": "Fundraising initiatives created by JPMorgan for issuers",
                "details": [
                    "Container for all bond issuances for specific capital raise",
                    "Each deal belongs to one issuer, contains multiple tranches",
                    "Coordinated by JPMorgan syndicate team"
                ],
                "parent_domains": ["ISSUER"],
                "child_domains": ["TRANCHE"],
                "key_concepts": [
                    "deal", "fundraising", "initiative", "capital_raise", 
                    "bond_issuance", "program", "offering", "mandates"
                ],
                "business_rules": [
                    "Each deal belongs to exactly one issuer",
                    "Deal must have at least one tranche",
                    "Deal terms define overall structure for tranches"
                ],
                "typical_queries": [
                    "deal summary and status",
                    "deals for specific issuer",
                    "deal performance metrics"
                ]
            },
            
            "TRANCHE": {
                "description": "Individual bond issuances with distinct terms",
                "details": [
                    "Core object with pricing, maturity, ratings information",
                    "Multiple tranches per deal allow different risk/return profiles",
                    "Each tranche belongs to one deal"
                ],
                "parent_domains": ["DEAL"],
                "child_domains": ["ORDER", "SYNDICATE"],
                "key_concepts": [
                    "tranche", "bond", "pricing", "maturity", "ratings", "yield", 
                    "spread", "financial_metrics", "terms", "conditions", "series"
                ],
                "business_rules": [
                    "Each tranche belongs to exactly one deal",
                    "Tranche pricing determines investor interest",
                    "Multiple tranches allow risk/return diversification"
                ],
                "typical_queries": [
                    "tranche pricing and yield information",
                    "tranche performance analysis",
                    "tranches by deal or issuer"
                ]
            },
            
            "SYNDICATE": {
                "description": "Financial institutions participating in distribution",
                "details": [
                    "Multiple banks per tranche with different roles (lead, co-manager)",
                    "Handle distribution and allocation decisions",
                    "Coordinate investor outreach and order management"
                ],
                "parent_domains": ["TRANCHE"],
                "child_domains": [],
                "key_concepts": [
                    "syndicate", "bank", "distribution", "lead_manager", "co_manager",
                    "underwriter", "bookrunner", "placement", "allocation"
                ],
                "business_rules": [
                    "Each tranche has a syndicate structure",
                    "Lead managers coordinate distribution",
                    "Allocation decisions made by syndicate banks"
                ],
                "typical_queries": [
                    "syndicate bank participation",
                    "allocation decisions by bank",
                    "lead manager performance"
                ]
            },
            
            "ORDER": {
                "description": "Investment requests from institutional investors",
                "details": [
                    "Contains IOI (Indication of Interest) and Final Allocation",
                    "Multiple orders per tranche from different investors",
                    "Foundation for allocation decisions"
                ],
                "parent_domains": ["TRANCHE"],
                "child_domains": ["ORDER_LIMIT"],
                "key_concepts": [
                    "order", "ioi", "indication_of_interest", "final_allocation", 
                    "investor", "investment", "demand", "interest", "booking"
                ],
                "business_rules": [
                    "Each order belongs to one tranche and one investor",
                    "IOI shows initial interest, allocation shows final amount",
                    "Final allocation often differs from IOI based on distribution strategy"
                ],
                "typical_queries": [
                    "order allocation analysis",
                    "IOI vs final allocation comparison",
                    "investor order patterns"
                ]
            },
            
            "ORDER_LIMIT": {
                "description": "Bond order amount within orders",
                "details": [
                    "Reoffer Order Limit: Unconditional investment amount",
                    "Conditional Order Limit: Investment with price/yield thresholds",
                    "Components that make up total order amounts"
                ],
                "parent_domains": ["ORDER"],
                "child_domains": [],
                "key_concepts": [
                    "order_limit", "reoffer", "conditional", "investment_amount", 
                    "threshold", "unconditional", "limit_type", "amount_component"
                ],
                "business_rules": [
                    "Reoffer limits are unconditional investment commitments",
                    "Conditional limits depend on final pricing terms",
                    "Total order = sum of all order limits"
                ],
                "typical_queries": [
                    "reoffer vs conditional amounts",
                    "order limit breakdown",
                    "unconditional investment analysis"
                ]
            },
            
            "INVESTOR": {
                "description": "Primary market investor entity that invests in deals",
                "details": [
                    "Places bond or hedge orders on tranches",
                    "Final investment given to investor is determined by syndicate bank allocation",
                    "Institutional investors with specific investment mandates"
                ],
                "parent_domains": [],
                "child_domains": ["ORDER"],
                "key_concepts": [
                    "investor", "institutional", "investment", "portfolio", 
                    "allocation", "mandate", "fund", "asset_manager", "pension"
                ],
                "business_rules": [
                    "Investors place orders for specific tranches",
                    "Investment allocation determined by syndicate banks",
                    "Investor profile affects allocation priority"
                ],
                "typical_queries": [
                    "investor portfolio analysis",
                    "allocation by investor type",
                    "investor participation patterns"
                ]
            },
            
            "TRADES": {
                "description": "Final record of actual trade execution",
                "details": [
                    "Contains trade date, price, yield, and other execution details",
                    "Trades can be of different types like investor trade, issuer trade, dealer trade",
                    "Settlement and post-trade processing information"
                ],
                "parent_domains": ["ORDER"],
                "child_domains": [],
                "key_concepts": [
                    "trade", "execution", "trade_date", "price", "yield", 
                    "dealer", "settlement", "confirmation", "clearing"
                ],
                "business_rules": [
                    "Trades represent final executed transactions",
                    "Trade details must match allocation decisions",
                    "Settlement follows trade execution"
                ],
                "typical_queries": [
                    "trade execution analysis",
                    "settlement status tracking",
                    "trade performance metrics"
                ]
            }
        }
    
    def create_entity_hierarchy_document(self) -> Dict[str, Any]:
        """Create the main entity hierarchy document."""
        
        hierarchy_doc = {
            "document_type": "BUSINESS_DOMAIN",
            "title": "Fixed Income Syndication Entity Hierarchy",
            "description": "Complete business domain hierarchy for fixed income syndication platform",
            "hierarchy_overview": "ISSUER → DEAL → TRANCHE → (ORDER & SYNDICATE) → ORDER_LIMIT & TRADES",
            "domains": self.business_hierarchy,
            "domain_relationships": self._create_relationship_matrix(),
            "query_routing_rules": self._create_query_routing_rules(),
            "metadata": {
                "created_at": "2024-01-01",  # Will be updated when script runs
                "document_type": "BUSINESS_DOMAIN",
                "total_domains": len(self.business_hierarchy),
                "hierarchy_levels": self._calculate_hierarchy_levels()
            }
        }
        
        return hierarchy_doc
    
    def _create_relationship_matrix(self) -> Dict[str, Dict[str, str]]:
        """Create a matrix showing relationships between domains."""
        relationships = {}
        
        for domain_name, domain_info in self.business_hierarchy.items():
            relationships[domain_name] = {
                "parents": domain_info.get("parent_domains", []),
                "children": domain_info.get("child_domains", []),
                "relationship_type": self._determine_relationship_type(domain_name, domain_info)
            }
        
        return relationships
    
    def _determine_relationship_type(self, domain_name: str, domain_info: Dict[str, Any]) -> str:
        """Determine the type of relationships this domain has."""
        has_parents = bool(domain_info.get("parent_domains"))
        has_children = bool(domain_info.get("child_domains"))
        
        if not has_parents and has_children:
            return "root_entity"
        elif has_parents and has_children:
            return "intermediate_entity"
        elif has_parents and not has_children:
            return "leaf_entity"
        else:
            return "standalone_entity"
    
    def _create_query_routing_rules(self) -> Dict[str, List[str]]:
        """Create rules for routing queries to appropriate domains."""
        routing_rules = {}
        
        for domain_name, domain_info in self.business_hierarchy.items():
            key_concepts = domain_info.get("key_concepts", [])
            typical_queries = domain_info.get("typical_queries", [])
            
            # Create routing patterns
            routing_patterns = []
            routing_patterns.extend(key_concepts)
            
            # Add query pattern keywords
            for query in typical_queries:
                query_words = query.lower().split()
                routing_patterns.extend([word for word in query_words if len(word) > 3])
            
            routing_rules[domain_name] = list(set(routing_patterns))  # Remove duplicates
        
        return routing_rules
    
    def _calculate_hierarchy_levels(self) -> Dict[str, int]:
        """Calculate the hierarchy level of each domain."""
        levels = {}
        
        # Start with root entities (no parents)
        for domain_name, domain_info in self.business_hierarchy.items():
            if not domain_info.get("parent_domains"):
                levels[domain_name] = 1
        
        # Calculate levels iteratively
        max_iterations = 10
        for iteration in range(max_iterations):
            changes_made = False
            
            for domain_name, domain_info in self.business_hierarchy.items():
                if domain_name in levels:
                    continue
                
                parent_domains = domain_info.get("parent_domains", [])
                if parent_domains and all(parent in levels for parent in parent_domains):
                    # Set level to max parent level + 1
                    max_parent_level = max(levels[parent] for parent in parent_domains)
                    levels[domain_name] = max_parent_level + 1
                    changes_made = True
            
            if not changes_made:
                break
        
        # Set remaining domains to level 1 (fallback)
        for domain_name in self.business_hierarchy:
            if domain_name not in levels:
                levels[domain_name] = 1
        
        return levels
    
    def create_individual_domain_documents(self) -> List[Dict[str, Any]]:
        """Create individual documents for each business domain."""
        domain_documents = []
        
        for domain_name, domain_info in self.business_hierarchy.items():
            doc = {
                "document_type": "BUSINESS_DOMAIN",
                "domain_name": domain_name,
                "title": f"Business Domain: {domain_name}",
                "description": domain_info["description"],
                "details": domain_info.get("details", []),
                "parent_domains": domain_info.get("parent_domains", []),
                "child_domains": domain_info.get("child_domains", []),
                "key_concepts": domain_info.get("key_concepts", []),
                "business_rules": domain_info.get("business_rules", []),
                "typical_queries": domain_info.get("typical_queries", []),
                "related_entities": self._get_related_entities(domain_name),
                "query_examples": self._generate_query_examples(domain_name, domain_info),
                "metadata": {
                    "created_at": "2024-01-01",  # Will be updated when script runs
                    "document_type": "BUSINESS_DOMAIN",
                    "domain_level": self._get_domain_level(domain_name),
                    "relationship_type": self._determine_relationship_type(domain_name, domain_info)
                }
            }
            
            domain_documents.append(doc)
        
        return domain_documents
    
    def _get_related_entities(self, domain_name: str) -> List[str]:
        """Get entities related to this domain (siblings, etc.)."""
        domain_info = self.business_hierarchy[domain_name]
        related = set()
        
        # Add parents and children
        related.update(domain_info.get("parent_domains", []))
        related.update(domain_info.get("child_domains", []))
        
        # Add siblings (entities with same parent)
        for parent in domain_info.get("parent_domains", []):
            if parent in self.business_hierarchy:
                related.update(self.business_hierarchy[parent].get("child_domains", []))
        
        # Remove self
        related.discard(domain_name)
        
        return sorted(list(related))
    
    def _get_domain_level(self, domain_name: str) -> int:
        """Get the hierarchy level of a domain."""
        levels = self._calculate_hierarchy_levels()
        return levels.get(domain_name, 1)
    
    def _generate_query_examples(self, domain_name: str, domain_info: Dict[str, Any]) -> List[str]:
        """Generate example queries for this domain."""
        examples = []
        
        # Use typical queries from domain info
        typical_queries = domain_info.get("typical_queries", [])
        examples.extend(typical_queries)
        
        # Generate additional examples based on key concepts
        key_concepts = domain_info.get("key_concepts", [])[:3]  # Take first 3 concepts
        
        for concept in key_concepts:
            examples.append(f"Show me all {concept} information")
            examples.append(f"Analyze {concept} performance")
        
        return examples[:5]  # Limit to 5 examples
    
    def create_business_rules_document(self) -> Dict[str, Any]:
        """Create a comprehensive business rules document."""
        
        all_rules = []
        rules_by_domain = {}
        
        for domain_name, domain_info in self.business_hierarchy.items():
            domain_rules = domain_info.get("business_rules", [])
            rules_by_domain[domain_name] = domain_rules
            all_rules.extend([f"{domain_name}: {rule}" for rule in domain_rules])
        
        rules_doc = {
            "document_type": "BUSINESS_DOMAIN",
            "title": "Fixed Income Syndication Business Rules",
            "description": "Comprehensive business rules governing entity relationships and operations",
            "all_business_rules": all_rules,
            "rules_by_domain": rules_by_domain,
            "cross_domain_rules": [
                "Entity hierarchy must be maintained: ISSUER → DEAL → TRANCHE",
                "All financial metrics roll up through the hierarchy",
                "Status changes propagate to related entities",
                "Allocation decisions affect multiple entity types",
                "Audit trails required for all entity modifications"
            ],
            "metadata": {
                "created_at": "2024-01-01",  # Will be updated when script runs
                "document_type": "BUSINESS_DOMAIN",
                "total_rules": len(all_rules),
                "domains_covered": len(rules_by_domain)
            }
        }
        
        return rules_doc
    
    def process_all_domains(self):
        """Main method to process all business domains."""
        logger.info("Starting business domain processing...")
        
        # Create entity hierarchy document
        hierarchy_doc = self.create_entity_hierarchy_document()
        hierarchy_file = self.business_domains_dir / "entity_hierarchy.json"
        with open(hierarchy_file, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_doc, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Created entity hierarchy: {hierarchy_file}")
        
        # Create individual domain documents
        domain_documents = self.create_individual_domain_documents()
        for doc in domain_documents:
            domain_name = doc["domain_name"].lower()
            domain_file = self.business_domains_dir / f"{domain_name}_domain.json"
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Created domain document: {domain_name}")
        
        # Create business rules document
        rules_doc = self.create_business_rules_document()
        rules_file = self.business_domains_dir / "business_rules.json"
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules_doc, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Created business rules: {rules_file}")
        
        # Create summary
        summary = {
            "processing_completed": True,
            "timestamp": "2024-01-01",  # Will be updated when script runs
            "statistics": {
                "domains_processed": len(domain_documents),
                "total_business_rules": len(rules_doc["all_business_rules"]),
                "hierarchy_levels": max(self._calculate_hierarchy_levels().values()),
                "total_key_concepts": sum(len(d.get("key_concepts", [])) for d in self.business_hierarchy.values())
            },
            "files_created": {
                "entity_hierarchy": str(hierarchy_file),
                "domain_documents": [f"{d['domain_name'].lower()}_domain.json" for d in domain_documents],
                "business_rules": str(rules_file)
            },
            "business_hierarchy": "ISSUER → DEAL → TRANCHE → (ORDER & SYNDICATE) → ORDER_LIMIT & TRADES",
            "notes": [
                "Business domain hierarchy based on fixed income syndication expertise",
                "Entity relationships encode real-world business processes",
                "Query routing rules support intelligent domain identification",
                "Business rules ensure data consistency and process compliance"
            ]
        }
        
        summary_file = self.output_dir / "business_domain_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Business domain processing complete:")
        logger.info(f"  ✅ Domains processed: {len(domain_documents)}")
        logger.info(f"  📋 Files created: {len(domain_documents) + 2}")
        logger.info(f"  📂 Output directory: {self.business_domains_dir}")
        logger.info(f"  📝 Summary: {summary_file}")
        
        return summary


def main():
    """Main function to run business domain processing."""
    
    # Configuration
    OUTPUT_DIR = "meta_documents/p1-synd/processed_domains"
    
    try:
        processor = BusinessDomainProcessor(OUTPUT_DIR)
        summary = processor.process_all_domains()
        
        print("\\n🏗️ Business Domain Processing Complete!")
        print("Files created:")
        print(f"  - Entity hierarchy: {OUTPUT_DIR}/business_domains/entity_hierarchy.json")
        print(f"  - Individual domains: {OUTPUT_DIR}/business_domains/*_domain.json")
        print(f"  - Business rules: {OUTPUT_DIR}/business_domains/business_rules.json")
        print(f"  - Summary: {OUTPUT_DIR}/business_domain_summary.json")
        print("\\nNext steps:")
        print("  1. Review generated business domain documents")
        print("  2. Update document sync service to process new domain structure")
        print("  3. Test the cascading RAG system with business domain identification")
        print("  4. Verify entity hierarchy matches business requirements")
        
    except Exception as e:
        logger.error(f"Business domain processing failed: {e}")


if __name__ == "__main__":
    print("🏗️ Business Domain Processing Script")
    print("=====================================")
    print("This script creates structured business domain hierarchy definitions")
    print("based on fixed income syndication business expertise.")
    print()
    
    # Ask for confirmation  
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled by user.")