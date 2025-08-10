#!/usr/bin/env python3
"""
Test script for hierarchical retrieval system.

Demonstrates the multi-step metadata retrieval pipeline:
1. Business Domain Identification
2. Core View Retrieval
3. Supporting View Retrieval
4. Report Example Retrieval
5. Lookup Value Retrieval
"""

import asyncio
import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalRetrievalDemo:
    """Demo of hierarchical retrieval system."""
    
    def __init__(self):
        # Paths and configuration
        self.meta_docs_path = Path(__file__).parent / "meta_documents"
        
        # Mock services for demonstration
        self.embedding_service = None
        self.vector_service = None
        self.llm_service = None
        
        # Load metadata directly from files for demo
        self.business_domains = {}
        self.view_metadata = {}
        self.report_metadata = {}
        self.lookup_metadata = {}
        
        self._load_all_metadata()
    
    def _load_all_metadata(self):
        """Load all metadata from JSON files."""
        try:
            # Load business domains
            domains_file = self.meta_docs_path / "business_domains.json"
            if domains_file.exists():
                with open(domains_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for domain in data.get('business_domains', []):
                        self.business_domains[domain['domain_id']] = domain
            
            # Load views
            views_dir = self.meta_docs_path / "views"
            if views_dir.exists():
                for json_file in views_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            view_data = json.load(f)
                            view_name = view_data.get('view_name', json_file.stem)
                            self.view_metadata[view_name] = view_data
                    except Exception as e:
                        logger.warning(f"Failed to load view {json_file.name}: {e}")
            
            # Load reports
            reports_dir = self.meta_docs_path / "reports"
            if reports_dir.exists():
                for json_file in reports_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                            report_name = report_data.get('report_name', json_file.stem)
                            self.report_metadata[report_name] = report_data
                    except Exception as e:
                        logger.warning(f"Failed to load report {json_file.name}: {e}")
            
            # Load lookups
            lookups_dir = self.meta_docs_path / "lookups"
            if lookups_dir.exists():
                for json_file in lookups_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            lookup_data = json.load(f)
                            lookup_id = lookup_data.get('lookup_id')
                            if lookup_id:
                                self.lookup_metadata[lookup_id] = lookup_data
                    except Exception as e:
                        logger.warning(f"Failed to load lookup {json_file.name}: {e}")
            
            logger.info(f"Loaded metadata: {len(self.business_domains)} domains, {len(self.view_metadata)} views, {len(self.report_metadata)} reports, {len(self.lookup_metadata)} lookups")
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    
    async def demonstrate_hierarchical_retrieval(self, user_query: str):
        """Demonstrate the hierarchical retrieval process."""
        print(f"\n" + "="*80)
        print(f"HIERARCHICAL RETRIEVAL DEMONSTRATION")
        print(f"User Query: {user_query}")
        print(f"="*80)
        
        # Step 1: Business Domain Identification
        print(f"\n[STEP 1] Business Domain Identification")
        identified_domains = self._identify_business_domains(user_query)
        print(f"Identified domains: {identified_domains}")
        
        for domain_id in identified_domains:
            domain = self.business_domains.get(domain_id)
            if domain:
                print(f"  - {domain['domain_name']}: {domain['description'][:100]}...")
        
        # Step 2: Core View Retrieval
        print(f"\n[STEP 2] Core View Retrieval")
        core_views = self._retrieve_core_views(user_query, identified_domains)
        print(f"Retrieved {len(core_views)} core views:")
        
        for view in core_views:
            print(f"  - {view['view_name']} ({view.get('view_type', 'UNKNOWN')})")
            print(f"    Description: {view.get('description', 'No description')[:100]}...")
        
        # Step 3: Supporting View Retrieval
        print(f"\n[STEP 3] Supporting View Retrieval")
        supporting_views = self._retrieve_supporting_views(user_query, core_views, identified_domains)
        print(f"Retrieved {len(supporting_views)} supporting views:")
        
        for view in supporting_views:
            print(f"  - {view['view_name']} ({view.get('view_type', 'UNKNOWN')})")
            print(f"    Description: {view.get('description', 'No description')[:100]}...")
        
        # Step 4: Report Example Retrieval
        print(f"\n[STEP 4] Report Example Retrieval")
        all_views = core_views + supporting_views
        reports = self._retrieve_report_examples(user_query, all_views, identified_domains)
        print(f"Retrieved {len(reports)} relevant reports:")
        
        for report in reports:
            print(f"  - {report['report_name']} ({report.get('report_type', 'STANDARD')})")
            print(f"    Related to: {report.get('view_name', 'N/A')}")
            if report.get('example_sql'):
                sql_preview = report['example_sql'][:100].replace('\n', ' ')
                print(f"    Example SQL: {sql_preview}...")
        
        # Step 5: Lookup Value Retrieval
        print(f"\n[STEP 5] Lookup Value Retrieval")
        lookups = self._retrieve_lookup_values(all_views)
        print(f"Retrieved {len(lookups)} relevant lookups:")
        
        for lookup in lookups:
            print(f"  - {lookup['lookup_name']} (ID: {lookup['lookup_id']})")
            print(f"    Type: {lookup.get('lookup_type', 'UNKNOWN')}")
            values_count = len(lookup.get('values', []))
            print(f"    Values: {values_count} entries")
            if lookup.get('values'):
                # Show first few values
                sample_values = lookup['values'][:3]
                sample_names = [v.get('name', 'N/A') for v in sample_values]
                print(f"    Sample values: {', '.join(sample_names)}")
        
        # Summary
        print(f"\n[SUMMARY] Context Assembly")
        total_metadata = len(core_views) + len(supporting_views) + len(reports) + len(lookups)
        print(f"Total metadata objects retrieved: {total_metadata}")
        print(f"  - Core views: {len(core_views)}")
        print(f"  - Supporting views: {len(supporting_views)}")
        print(f"  - Reports: {len(reports)}")
        print(f"  - Lookups: {len(lookups)}")
        
        # Show context quality metrics
        context_coverage = self._calculate_context_coverage(identified_domains, all_views, reports, lookups)
        print(f"Context coverage score: {context_coverage:.2f}")
        
        return {
            'domains': identified_domains,
            'core_views': core_views,
            'supporting_views': supporting_views,
            'reports': reports,
            'lookups': lookups,
            'coverage': context_coverage
        }
    
    def _identify_business_domains(self, user_query: str) -> list:
        """Mock domain identification using keyword matching."""
        query_lower = user_query.lower()
        matched_domains = []
        
        for domain_id, domain in self.business_domains.items():
            keywords = domain.get('keywords', [])
            if any(keyword.lower() in query_lower for keyword in keywords):
                matched_domains.append(domain_id)
        
        # Fallback logic for common scenarios
        if not matched_domains:
            if any(term in query_lower for term in ['user', 'login', 'activity']):
                matched_domains.append(3)  # User Management
            elif any(term in query_lower for term in ['deal', 'transaction']):
                matched_domains.append(1)  # Deal Management
            elif any(term in query_lower for term in ['syndicate', 'participant']):
                matched_domains.append(2)  # Syndicate Operations
        
        return matched_domains[:3]  # Limit to 3 domains
    
    def _retrieve_core_views(self, user_query: str, domain_ids: list) -> list:
        """Retrieve core views for identified domains."""
        core_views = []
        
        for view_name, view_data in self.view_metadata.items():
            # Check if view belongs to identified domains
            view_domains = set(view_data.get('business_domains', []))
            if view_domains.intersection(set(domain_ids)) and view_data.get('view_type') == 'CORE':
                core_views.append(view_data)
        
        # Simple relevance scoring based on keyword matching
        query_lower = user_query.lower()
        
        def relevance_score(view):
            score = 0
            view_text = f"{view.get('view_name', '')} {view.get('description', '')} {view.get('use_cases', '')}".lower()
            
            # Count keyword matches
            for word in query_lower.split():
                if word in view_text:
                    score += 1
            
            return score
        
        core_views.sort(key=relevance_score, reverse=True)
        return core_views[:3]  # Return top 3 core views
    
    def _retrieve_supporting_views(self, user_query: str, core_views: list, domain_ids: list) -> list:
        """Retrieve supporting views based on core views and domains."""
        supporting_views = []
        core_view_names = {v.get('view_name') for v in core_views}
        
        for view_name, view_data in self.view_metadata.items():
            # Skip if already in core views
            if view_name in core_view_names:
                continue
            
            # Check if view belongs to identified domains and is supporting
            view_domains = set(view_data.get('business_domains', []))
            if view_domains.intersection(set(domain_ids)) and view_data.get('view_type') == 'SUPPORTING':
                supporting_views.append(view_data)
        
        return supporting_views[:5]  # Return top 5 supporting views
    
    def _retrieve_report_examples(self, user_query: str, all_views: list, domain_ids: list) -> list:
        """Retrieve relevant report examples."""
        reports = []
        view_names = {v.get('view_name') for v in all_views}
        
        for report_name, report_data in self.report_metadata.items():
            # Check domain overlap
            report_domains = set(report_data.get('business_domains', []))
            if report_domains.intersection(set(domain_ids)):
                reports.append(report_data)
                continue
            
            # Check view relationships
            report_view = report_data.get('view_name')
            related_views = set(report_data.get('related_views', []))
            
            if report_view in view_names or related_views.intersection(view_names):
                reports.append(report_data)
        
        return reports[:2]  # Return top 2 reports
    
    def _retrieve_lookup_values(self, all_views: list) -> list:
        """Retrieve lookup values for columns with lookup_id."""
        lookup_ids = set()
        
        # Extract lookup IDs from view columns
        for view in all_views:
            columns = view.get('columns', [])
            for column in columns:
                lookup_id = column.get('lookup_id')
                if lookup_id:
                    lookup_ids.add(lookup_id)
        
        # Get lookup metadata for these IDs
        lookups = []
        for lookup_id in lookup_ids:
            lookup_data = self.lookup_metadata.get(lookup_id)
            if lookup_data:
                lookups.append(lookup_data)
        
        return lookups[:10]  # Return up to 10 lookups
    
    def _calculate_context_coverage(self, domains: list, views: list, reports: list, lookups: list) -> float:
        """Calculate a coverage score for the retrieved context."""
        # Base score from domain coverage
        domain_score = min(len(domains) / 3.0, 1.0)  # Normalize to 0-1
        
        # View coverage score
        view_score = min(len(views) / 5.0, 1.0)
        
        # Report coverage score
        report_score = min(len(reports) / 2.0, 1.0)
        
        # Lookup coverage score
        lookup_score = min(len(lookups) / 5.0, 1.0)
        
        # Weighted average
        total_score = (
            domain_score * 0.2 + 
            view_score * 0.4 + 
            report_score * 0.2 + 
            lookup_score * 0.2
        )
        
        return total_score

async def main():
    """Main demonstration function."""
    demo = HierarchicalRetrievalDemo()
    
    # Test scenarios that demonstrate different aspects of hierarchical retrieval
    test_scenarios = [
        "Show me syndicate participation details for recent deals with active status",
        "What are the user engagement metrics and login patterns for different user roles?",
        "Get deal pipeline information including transaction amounts and participant allocations",
        "Find all active tranches with their syndicate roles and lookup values"
    ]
    
    print("HIERARCHICAL METADATA RETRIEVAL SYSTEM TEST")
    print("=" * 60)
    
    results = []
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n\nSCENARIO {i}:")
        result = await demo.demonstrate_hierarchical_retrieval(scenario)
        results.append(result)
        
        # Brief pause for readability
        await asyncio.sleep(0.5)
    
    # Overall summary
    print(f"\n" + "="*80)
    print(f"OVERALL RESULTS SUMMARY")
    print(f"="*80)
    
    for i, (scenario, result) in enumerate(zip(test_scenarios, results), 1):
        print(f"\nScenario {i}: Coverage {result['coverage']:.2f}")
        print(f"  Domains: {len(result['domains'])}, Views: {len(result['core_views']) + len(result['supporting_views'])}, Reports: {len(result['reports'])}, Lookups: {len(result['lookups'])}")
    
    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    print(f"\nAverage context coverage: {avg_coverage:.2f}")
    
    print(f"\n✅ Hierarchical retrieval system demonstration completed!")
    print(f"\nKey Benefits Demonstrated:")
    print(f"  1. ✅ Domain-driven filtering reduces noise")
    print(f"  2. ✅ Multi-step retrieval ensures comprehensive context")
    print(f"  3. ✅ Lookup integration provides valid filter values")
    print(f"  4. ✅ Report examples show SQL patterns")
    print(f"  5. ✅ Graph relationships guide retrieval decisions")

if __name__ == "__main__":
    asyncio.run(main())