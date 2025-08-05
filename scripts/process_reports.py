#!/usr/bin/env python3
"""
Report Processing Script for Business Domain-First Architecture.
Processes individual report JSON files with variable keys and links them to relevant views.
Creates structured content for RAG consumption with query patterns and examples.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportProcessor:
    """Process individual report JSON files into structured RAG-ready content."""
    
    def __init__(self, reports_dir: str, output_dir: str):
        """
        Initialize report processor.
        
        Args:
            reports_dir: Directory containing individual report JSON files
            output_dir: Directory to save processed report files
        """
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.processed_reports_dir = self.output_dir / "processed_reports"
        
        # Create directories
        self.processed_reports_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.reports_dir.exists():
            raise FileNotFoundError(f"Reports directory not found: {reports_dir}")
        
        # Business domain assignment rules
        self.domain_assignment_rules = {
            "issuer": ["ISSUER"],
            "company": ["ISSUER"],
            "deal": ["DEAL", "ISSUER"],
            "fundrais": ["DEAL", "ISSUER"],
            "tranche": ["TRANCHE", "DEAL"],
            "bond": ["TRANCHE", "DEAL"],
            "pricing": ["TRANCHE"],
            "yield": ["TRANCHE"],
            "syndicate": ["SYNDICATE", "TRANCHE"],
            "bank": ["SYNDICATE"],
            "order": ["ORDER", "TRANCHE"],
            "allocation": ["ORDER", "INVESTOR", "SYNDICATE"],
            "ioi": ["ORDER", "INVESTOR"],
            "investor": ["INVESTOR", "ORDER"],
            "trade": ["TRADES", "ORDER"],
            "execution": ["TRADES"],
            "settlement": ["TRADES"],
            "user": ["USER", "SYSTEM"],
            "performance": ["SYSTEM"],
            "metrics": ["SYSTEM"],
            "analysis": ["SYSTEM"]
        }
        
        # Expected report keys (some may be missing in actual files)
        self.expected_keys = [
            "view_name", "view_sql", "name", "report_description", 
            "data_returned", "example_sql", "use_cases"
        ]
    
    def find_report_files(self) -> List[Path]:
        """Find all JSON report files in the reports directory."""
        report_files = []
        
        # Look for JSON files
        for json_file in self.reports_dir.glob("*.json"):
            report_files.append(json_file)
        
        logger.info(f"Found {len(report_files)} report files")
        return report_files
    
    def load_report(self, file_path: Path) -> Dict[str, Any]:
        """Load a single report JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to load report {file_path}: {e}")
            return {}
    
    def assign_business_domains(self, report_data: Dict[str, Any]) -> List[str]:
        """Assign business domains based on report content."""
        # Collect text for analysis
        text_fields = []
        
        for key in ["name", "report_description", "use_cases", "view_name"]:
            value = report_data.get(key, "")
            if isinstance(value, str):
                text_fields.append(value.lower())
            elif isinstance(value, list):
                text_fields.extend([str(item).lower() for item in value])
        
        combined_text = " ".join(text_fields)
        
        assigned_domains = set()
        
        # Check assignment rules
        for pattern, domains in self.domain_assignment_rules.items():
            if pattern in combined_text:
                assigned_domains.update(domains)
        
        # If no specific assignment, default to SYSTEM
        if not assigned_domains:
            assigned_domains.add("SYSTEM")
        
        return sorted(list(assigned_domains))
    
    def extract_related_views(self, report_data: Dict[str, Any]) -> List[str]:
        """Extract views mentioned in the report."""
        related_views = []
        
        # Direct view_name reference
        view_name = report_data.get("view_name")
        if view_name:
            related_views.append(view_name.upper())
        
        # Extract views from SQL content
        sql_fields = ["view_sql", "example_sql"]
        for field in sql_fields:
            sql_content = report_data.get(field, "")
            if isinstance(sql_content, str):
                # Find view references in SQL (pattern: FROM view_name or JOIN view_name)
                view_matches = re.findall(r'(?:FROM|JOIN)\s+(\w*V_\w+)', sql_content, re.IGNORECASE)
                for match in view_matches:
                    if match.upper() not in related_views:
                        related_views.append(match.upper())
        
        # Extract from description text
        description = report_data.get("report_description", "")
        if isinstance(description, str):
            view_matches = re.findall(r'\bV_\w+\b', description, re.IGNORECASE)
            for match in view_matches:
                if match.upper() not in related_views:
                    related_views.append(match.upper())
        
        return related_views
    
    def extract_query_patterns(self, report_data: Dict[str, Any]) -> List[str]:
        """Extract SQL query patterns from the report."""
        patterns = []
        
        # Extract from example_sql
        example_sql = report_data.get("example_sql")
        if example_sql:
            cleaned_sql = self._clean_sql_pattern(example_sql)
            if cleaned_sql:
                patterns.append(cleaned_sql)
        
        # Extract from view_sql (if it's a query example)
        view_sql = report_data.get("view_sql")
        if view_sql and "SELECT" in view_sql.upper():
            cleaned_sql = self._clean_sql_pattern(view_sql)
            if cleaned_sql and cleaned_sql not in patterns:
                patterns.append(cleaned_sql)
        
        # Extract SQL patterns from description
        description = report_data.get("report_description", "")
        if isinstance(description, str):
            sql_matches = re.findall(r'SELECT.*?(?:;|$)', description, re.IGNORECASE | re.DOTALL)
            for match in sql_matches:
                cleaned_sql = self._clean_sql_pattern(match)
                if cleaned_sql and cleaned_sql not in patterns:
                    patterns.append(cleaned_sql)
        
        return patterns[:3]  # Limit to 3 patterns
    
    def _clean_sql_pattern(self, sql: str) -> str:
        """Clean and format SQL pattern for better readability."""
        if not sql:
            return ""
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', sql.strip())
        
        # Remove comments
        cleaned = re.sub(r'--.*?$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # Basic SQL formatting
        cleaned = cleaned.replace(' FROM ', '\nFROM ')
        cleaned = cleaned.replace(' WHERE ', '\nWHERE ')
        cleaned = cleaned.replace(' JOIN ', '\nJOIN ')
        cleaned = cleaned.replace(' ORDER BY ', '\nORDER BY ')
        cleaned = cleaned.replace(' GROUP BY ', '\nGROUP BY ')
        
        return cleaned.strip()
    
    def process_report(self, file_path: Path, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single report into enhanced metadata."""
        
        # Extract basic information
        report_name = report_data.get("name") or file_path.stem
        
        # Assign business domains
        business_domains = self.assign_business_domains(report_data)
        
        # Extract related views
        related_views = self.extract_related_views(report_data)
        
        # Extract use cases
        use_cases = report_data.get("use_cases", [])
        if isinstance(use_cases, str):
            # Split string use cases by line breaks or bullet points
            use_cases = [case.strip("- ").strip() for case in use_cases.split("\n") if case.strip()]
        elif not isinstance(use_cases, list):
            use_cases = []
        
        # Extract query patterns
        query_patterns = self.extract_query_patterns(report_data)
        
        # Create enhanced metadata
        enhanced_metadata = {
            "name": report_name,
            "report_description": report_data.get("report_description", ""),
            "business_domains": business_domains,
            "related_views": related_views,
            "use_cases": use_cases[:5],  # Limit to 5 use cases
            "data_returned": report_data.get("data_returned"),
            "example_sql": report_data.get("example_sql"),
            "view_name": report_data.get("view_name"),
            "view_sql": report_data.get("view_sql"),
            "query_patterns": query_patterns,
            "original_keys": list(report_data.keys()),  # Track what was in original
            "metadata": {
                "processed_at": "2024-01-01",  # Will be updated when script runs
                "source_file": file_path.name,
                "document_type": "REPORT",
                "has_view_link": bool(report_data.get("view_name")),
                "has_sql_examples": bool(report_data.get("example_sql") or report_data.get("view_sql"))
            }
        }
        
        return enhanced_metadata
    
    def create_report_catalog(self, processed_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a catalog of all processed reports."""
        
        # Group reports by business domain
        domain_groups = {}
        for report in processed_reports:
            for domain in report.get("business_domains", []):
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append({
                    "name": report["name"],
                    "description": report["report_description"][:100] + "..." if len(report["report_description"]) > 100 else report["report_description"],
                    "related_views": report["related_views"],
                    "has_sql": report["metadata"]["has_sql_examples"]
                })
        
        # Collect view linkages
        view_linkages = {}
        for report in processed_reports:
            for view in report.get("related_views", []):
                if view not in view_linkages:
                    view_linkages[view] = []
                view_linkages[view].append(report["name"])
        
        # Create catalog
        catalog = {
            "catalog_created": True,
            "timestamp": "2024-01-01",  # Will be updated when script runs
            "statistics": {
                "total_reports": len(processed_reports),
                "reports_with_views": sum(1 for r in processed_reports if r["metadata"]["has_view_link"]),
                "reports_with_sql": sum(1 for r in processed_reports if r["metadata"]["has_sql_examples"]),
                "business_domains_covered": len(domain_groups),
                "total_view_linkages": len(view_linkages)
            },
            "reports_by_domain": domain_groups,
            "view_linkages": view_linkages,
            "available_query_patterns": sum(len(r.get("query_patterns", [])) for r in processed_reports),
            "notes": [
                "Reports processed from individual JSON files",
                "Business domains auto-assigned based on content analysis",
                "View linkages extracted from view_name and SQL content",
                "Query patterns cleaned and formatted for RAG consumption"
            ]
        }
        
        return catalog
    
    def process_all_reports(self):
        """Main method to process all report files."""
        logger.info("Starting report processing...")
        
        # Find all report files
        report_files = self.find_report_files()
        if not report_files:
            logger.warning("No report files found")
            return
        
        processed_reports = []
        failed_reports = 0
        
        for file_path in report_files:
            try:
                # Load report data
                report_data = self.load_report(file_path)
                if not report_data:
                    failed_reports += 1
                    continue
                
                # Process report
                enhanced_metadata = self.process_report(file_path, report_data)
                
                # Save processed report
                output_file = self.processed_reports_dir / f"{file_path.stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
                
                processed_reports.append(enhanced_metadata)
                logger.info(f"✅ Processed report: {enhanced_metadata['name']}")
                
            except Exception as e:
                failed_reports += 1
                logger.error(f"❌ Failed to process report {file_path}: {e}")
        
        # Create report catalog
        catalog = self.create_report_catalog(processed_reports)
        
        # Save catalog
        catalog_file = self.output_dir / "report_catalog.json"
        with open(catalog_file, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        
        # Create summary
        summary = {
            "processing_completed": True,
            "timestamp": "2024-01-01",  # Will be updated when script runs
            "statistics": {
                "reports_processed": len(processed_reports),
                "reports_failed": failed_reports,
                "total_files_found": len(report_files)
            },
            "output_files": {
                "processed_reports_dir": str(self.processed_reports_dir),
                "catalog_file": str(catalog_file)
            },
            "next_steps": [
                "Review report catalog for business domain assignments",
                "Verify view linkages are correct",
                "Run process_business_domains.py to create business hierarchy",
                "Update document sync service to process new report structure"
            ]
        }
        
        summary_file = self.output_dir / "report_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Processing complete:")
        logger.info(f"  ✅ Reports processed: {len(processed_reports)}")
        logger.info(f"  ❌ Reports failed: {failed_reports}")
        logger.info(f"  📂 Output directory: {self.output_dir}")
        logger.info(f"  📋 Catalog saved: {catalog_file}")
        
        return summary


def main():
    """Main function to run report processing."""
    
    # Configuration
    REPORTS_DIR = "meta_documents/p1-synd/reports"
    OUTPUT_DIR = "meta_documents/p1-synd/processed_reports"
    
    try:
        processor = ReportProcessor(REPORTS_DIR, OUTPUT_DIR)
        summary = processor.process_all_reports()
        
        print("\\n📋 Report Processing Complete!")
        print("Files created:")
        print(f"  - Processed reports: {OUTPUT_DIR}/processed_reports/")
        print(f"  - Report catalog: {OUTPUT_DIR}/report_catalog.json")
        print(f"  - Summary: {OUTPUT_DIR}/report_processing_summary.json")
        print("\\nNext steps:")
        print("  1. Review report catalog for business domain assignments")
        print("  2. Verify view linkages are correctly identified")
        print("  3. Run process_business_domains.py to create business hierarchy")
        print("  4. Update document sync service for new metadata structure")
        
    except Exception as e:
        logger.error(f"Report processing failed: {e}")


if __name__ == "__main__":
    print("📋 Report Processing Script")
    print("===========================")
    print("This script processes individual report JSON files and creates")
    print("structured content for RAG consumption with query patterns.")
    print()
    
    # Ask for confirmation  
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled by user.")