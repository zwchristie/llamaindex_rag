#!/usr/bin/env python3
"""
View Metadata Processing Script for Business Domain-First Architecture.
Processes view_metadata.json containing CORE_VIEWS and SUPPORTING_VIEWS arrays.
Creates individual metadata files with business domain assignments and dependencies.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViewMetadataProcessor:
    """Process view_metadata.json into individual view files with business domain intelligence."""
    
    def __init__(self, input_file: str, output_dir: str):
        """
        Initialize view metadata processor.
        
        Args:
            input_file: Path to view_metadata.json
            output_dir: Directory to save processed view files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.core_views_dir = self.output_dir / "core_views"
        self.supporting_views_dir = self.output_dir / "supporting_views"
        
        # Create directories
        self.core_views_dir.mkdir(parents=True, exist_ok=True)
        self.supporting_views_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Business domain assignment rules based on user's hierarchy
        self.domain_assignment_rules = {
            # Core business entity patterns
            "issuer": ["ISSUER"],
            "company": ["ISSUER"],
            
            "deal": ["DEAL", "ISSUER"],
            "fundrais": ["DEAL", "ISSUER"],
            
            "tranche": ["TRANCHE", "DEAL"],
            "bond": ["TRANCHE", "DEAL"],
            "instrument": ["TRANCHE", "DEAL"],
            "pricing": ["TRANCHE"],
            "yield": ["TRANCHE"],
            "maturity": ["TRANCHE"],
            
            "syndicate": ["SYNDICATE", "TRANCHE"],
            "bank": ["SYNDICATE"],
            
            "order": ["ORDER", "TRANCHE"],
            "allocation": ["ORDER", "INVESTOR", "SYNDICATE"],
            "ioi": ["ORDER", "INVESTOR"],
            
            "investor": ["INVESTOR", "ORDER"],
            "institution": ["INVESTOR"],
            
            "trade": ["TRADES", "ORDER"],
            "execution": ["TRADES"],
            "settlement": ["TRADES"],
            
            # Supporting patterns
            "user": ["USER", "SYSTEM"],
            "metrics": ["SYSTEM"],
            "audit": ["SYSTEM"],
            "status": ["SYSTEM"],
            "lookup": ["SYSTEM"]
        }
    
    def load_view_metadata(self) -> Dict[str, Any]:
        """Load view metadata from JSON file."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded view metadata from {self.input_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load view metadata: {e}")
            raise
    
    def assign_business_domains(self, view_name: str, description: str = "") -> List[str]:
        """Assign business domains based on view name and description."""
        view_lower = view_name.lower()
        desc_lower = description.lower()
        combined_text = f"{view_lower} {desc_lower}"
        
        assigned_domains = set()
        
        # Check assignment rules
        for pattern, domains in self.domain_assignment_rules.items():
            if pattern in combined_text:
                assigned_domains.update(domains)
        
        # If no specific assignment, try to infer from view name patterns
        if not assigned_domains:
            if "v_" in view_lower:
                # Remove V_ prefix and analyze
                clean_name = view_lower.replace("v_", "").replace("_", " ")
                for pattern, domains in self.domain_assignment_rules.items():
                    if pattern in clean_name:
                        assigned_domains.update(domains)
        
        # Default assignment if nothing matches
        if not assigned_domains:
            assigned_domains.add("SYSTEM")
        
        return sorted(list(assigned_domains))
    
    def process_core_view(self, view_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a core view into enhanced metadata."""
        view_name = view_data.get("view_name", "")
        if not view_name:
            logger.warning("Core view missing view_name", view_data_keys=list(view_data.keys()))
            return None
        
        # Assign business domains
        description = view_data.get("description", "")
        business_domains = self.assign_business_domains(view_name, description)
        
        # Extract and clean use cases
        use_cases = view_data.get("use_cases", [])
        if isinstance(use_cases, str):
            # Split string use cases by line breaks or bullet points
            use_cases = [case.strip("- ").strip() for case in use_cases.split("\n") if case.strip()]
        elif not isinstance(use_cases, list):
            use_cases = []
        
        # Create enhanced metadata
        enhanced_metadata = {
            "view_name": view_name.upper(),
            "view_type": view_data.get("view_type", "Core view"),
            "description": description,
            "business_domains": business_domains,
            "data_returned": view_data.get("data_returned", ""),
            "use_cases": use_cases[:5],  # Limit to 5 use cases
            "example_query": view_data.get("example_query"),
            "view_sql": view_data.get("view_sql"),
            "financial_context": self._generate_financial_context(view_name, business_domains),
            "metadata": {
                "processed_at": "2024-01-01",  # Will be updated when script runs
                "source_file": "view_metadata.json",
                "document_type": "CORE_VIEW"
            }
        }
        
        return enhanced_metadata
    
    def process_supporting_view(self, view_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a supporting view into enhanced metadata."""
        view_name = view_data.get("view_name", "")
        if not view_name:
            logger.warning("Supporting view missing view_name", view_data_keys=list(view_data.keys()))
            return None
        
        # Assign business domains
        description = view_data.get("description", "")
        business_domains = self.assign_business_domains(view_name, description)
        
        # Extract and clean use cases
        use_cases = view_data.get("use_cases", [])
        if isinstance(use_cases, str):
            use_cases = [case.strip("- ").strip() for case in use_cases.split("\n") if case.strip()]
        elif not isinstance(use_cases, list):
            use_cases = []
        
        # Extract views supported
        views_supported = view_data.get("views_supported", [])
        if not isinstance(views_supported, list):
            views_supported = []
        
        # Create enhanced metadata
        enhanced_metadata = {
            "view_name": view_name.upper(),
            "view_type": view_data.get("view_type", "Supporting view"),
            "description": description,
            "business_domains": business_domains,
            "views_supported": [v.upper() for v in views_supported],
            "data_returned": view_data.get("data_returned", ""),
            "use_cases": use_cases[:5],  # Limit to 5 use cases
            "example_query": view_data.get("example_query"),
            "view_sql": view_data.get("view_sql"),
            "enhancement_provided": self._generate_enhancement_description(view_name, views_supported),
            "metadata": {
                "processed_at": "2024-01-01",  # Will be updated when script runs
                "source_file": "view_metadata.json",
                "document_type": "SUPPORTING_VIEW"
            }
        }
        
        return enhanced_metadata
    
    def _generate_financial_context(self, view_name: str, business_domains: List[str]) -> Optional[str]:
        """Generate financial domain context for core views."""
        view_lower = view_name.lower()
        
        financial_contexts = {
            "TRANCHE": "Fixed income tranche data with pricing, yield, and maturity information",
            "DEAL": "Bond deal information including issuance details and syndicate structure",
            "ORDER": "Investment order data with IOI, allocation, and investor information",
            "INVESTOR": "Institutional investor portfolio and investment activity",
            "TRADES": "Trade execution and settlement data for bond transactions",
            "SYNDICATE": "Syndicate bank participation and allocation responsibilities"
        }
        
        # Find most relevant financial context
        for domain in business_domains:
            if domain in financial_contexts:
                return financial_contexts[domain]
        
        # Generate based on view name patterns
        if any(term in view_lower for term in ["pricing", "yield", "spread"]):
            return "Financial pricing and yield information for bond valuation"
        elif any(term in view_lower for term in ["allocation", "ioi"]):
            return "Investment allocation and order management data"
        elif any(term in view_lower for term in ["trade", "execution"]):
            return "Trade execution and settlement information"
        
        return None
    
    def _generate_enhancement_description(self, view_name: str, views_supported: List[str]) -> Optional[str]:
        """Generate description of how this supporting view enhances core views."""
        if not views_supported:
            return None
        
        view_lower = view_name.lower()
        
        if "detail" in view_lower or "breakdown" in view_lower:
            return f"Provides detailed breakdown and additional context for {', '.join(views_supported)}"
        elif "instrument" in view_lower:
            return f"Provides instrument-level details and financial metrics for {', '.join(views_supported)}"
        elif "settlement" in view_lower:
            return f"Provides trade settlement and execution details for {', '.join(views_supported)}"
        else:
            return f"Provides supporting data and enhanced context for {', '.join(views_supported)}"
    
    def create_view_dependency_mapping(self, core_views: List[Dict], supporting_views: List[Dict]) -> Dict[str, List[str]]:
        """Create mapping of core views to their supporting views."""
        dependencies = {}
        
        # Create lookup of supporting views by what they support
        supporting_lookup = {}
        for supporting_view in supporting_views:
            view_name = supporting_view.get("view_name", "").upper()
            views_supported = supporting_view.get("views_supported", [])
            for core_view in views_supported:
                core_view_upper = core_view.upper()
                if core_view_upper not in supporting_lookup:
                    supporting_lookup[core_view_upper] = []
                supporting_lookup[core_view_upper].append(view_name)
        
        # Map core views to their supporting views
        for core_view in core_views:
            view_name = core_view.get("view_name", "").upper()
            if view_name in supporting_lookup:
                dependencies[view_name] = supporting_lookup[view_name]
        
        return dependencies
    
    def process_all_views(self):
        """Main method to process all views from view_metadata.json."""
        logger.info("Starting view metadata processing...")
        
        # Load metadata
        metadata = self.load_view_metadata()
        
        core_views_data = metadata.get("CORE_VIEWS", [])
        supporting_views_data = metadata.get("SUPPORTING_VIEWS", [])
        
        logger.info(f"Found {len(core_views_data)} core views and {len(supporting_views_data)} supporting views")
        
        # Process core views
        core_views_processed = []
        core_views_failed = 0
        
        for i, core_view_data in enumerate(core_views_data):
            try:
                enhanced_metadata = self.process_core_view(core_view_data)
                if enhanced_metadata:
                    view_name = enhanced_metadata["view_name"]
                    output_file = self.core_views_dir / f"{view_name.lower()}.json"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
                    
                    core_views_processed.append(enhanced_metadata)
                    logger.info(f"✅ Processed core view: {view_name}")
                else:
                    core_views_failed += 1
                    logger.error(f"❌ Failed to process core view {i}")
                    
            except Exception as e:
                core_views_failed += 1
                logger.error(f"❌ Failed to process core view {i}: {e}")
        
        # Process supporting views
        supporting_views_processed = []
        supporting_views_failed = 0
        
        for i, supporting_view_data in enumerate(supporting_views_data):
            try:
                enhanced_metadata = self.process_supporting_view(supporting_view_data)
                if enhanced_metadata:
                    view_name = enhanced_metadata["view_name"]
                    output_file = self.supporting_views_dir / f"{view_name.lower()}.json"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
                    
                    supporting_views_processed.append(enhanced_metadata)
                    logger.info(f"✅ Processed supporting view: {view_name}")
                else:
                    supporting_views_failed += 1
                    logger.error(f"❌ Failed to process supporting view {i}")
                    
            except Exception as e:
                supporting_views_failed += 1
                logger.error(f"❌ Failed to process supporting view {i}: {e}")
        
        # Create dependency mapping
        dependencies = self.create_view_dependency_mapping(core_views_processed, supporting_views_processed)
        
        # Create summary file
        summary = {
            "processing_completed": True,
            "timestamp": "2024-01-01",  # Will be updated when script runs
            "statistics": {
                "core_views_processed": len(core_views_processed),
                "core_views_failed": core_views_failed,
                "supporting_views_processed": len(supporting_views_processed),
                "supporting_views_failed": supporting_views_failed,
                "total_dependencies": len(dependencies)
            },
            "view_dependencies": dependencies,
            "business_domains_assigned": self._collect_assigned_domains(core_views_processed + supporting_views_processed),
            "notes": [
                "Core views saved in core_views/ directory",
                "Supporting views saved in supporting_views/ directory",
                "Each view assigned business domains based on naming patterns",
                "View dependencies mapped for hierarchical context service"
            ]
        }
        
        summary_file = self.output_dir / "view_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Processing complete:")
        logger.info(f"  ✅ Core views: {len(core_views_processed)} processed, {core_views_failed} failed")
        logger.info(f"  ✅ Supporting views: {len(supporting_views_processed)} processed, {supporting_views_failed} failed")
        logger.info(f"  📝 Summary saved: {summary_file}")
        
        return summary
    
    def _collect_assigned_domains(self, all_views: List[Dict]) -> Dict[str, int]:
        """Collect statistics on assigned business domains."""
        domain_counts = {}
        
        for view in all_views:
            domains = view.get("business_domains", [])
            for domain in domains:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        return domain_counts


def main():
    """Main function to run view metadata processing."""
    
    # Configuration
    INPUT_FILE = "meta_documents/p1-synd/views/view_metadata.json"
    OUTPUT_DIR = "meta_documents/p1-synd/processed_views"
    
    try:
        processor = ViewMetadataProcessor(INPUT_FILE, OUTPUT_DIR)
        summary = processor.process_all_views()
        
        print("\\n📋 View Metadata Processing Complete!")
        print("Files created:")
        print(f"  - Core views: {OUTPUT_DIR}/core_views/")
        print(f"  - Supporting views: {OUTPUT_DIR}/supporting_views/")
        print(f"  - Summary: {OUTPUT_DIR}/view_processing_summary.json")
        print("\\nNext steps:")
        print("  1. Review generated view files and business domain assignments")
        print("  2. Run process_reports.py to handle individual report JSON files")
        print("  3. Run process_business_domains.py to create business hierarchy")
        print("  4. Update document sync service to process new view structure")
        
    except Exception as e:
        logger.error(f"View processing failed: {e}")


if __name__ == "__main__":
    print("📋 View Metadata Processing Script")
    print("==================================")
    print("This script processes view_metadata.json with CORE_VIEWS and SUPPORTING_VIEWS")
    print("into individual view files with business domain assignments.")
    print()
    
    # Ask for confirmation  
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled by user.")