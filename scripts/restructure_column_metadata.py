#!/usr/bin/env python3
"""
Housekeeping script to restructure existing main_schema_metadata.json into new column metadata format.
This script will be used once to transform existing metadata to new tiered format.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ColumnMetadataRestructurer:
    """Restructure existing schema metadata into new column-focused format."""
    
    def __init__(self, input_file: str, output_dir: str):
        """
        Initialize restructurer.
        
        Args:
            input_file: Path to main_schema_metadata.json
            output_dir: Directory to save new column metadata files
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
    def load_existing_metadata(self) -> Dict[str, Any]:
        """Load existing schema metadata."""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded metadata from {self.input_file}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise
    
    def create_business_descriptions(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create business descriptions file from existing metadata."""
        
        # Extract catalog and schema info
        catalog = metadata.get("catalog", "unknown")
        schema_name = metadata.get("schema", "unknown")
        
        # Group tables by functional domains (you can customize this logic)
        domains = {
            "trading_lifecycle": {
                "description": "Core trading and order management tables",
                "tables": {}
            },
            "user_management": {
                "description": "User accounts and authentication",
                "tables": {}
            },
            "syndicate_operations": {
                "description": "Syndicate and tranche management",
                "tables": {}
            },
            "reporting": {
                "description": "Reporting and analytics tables",
                "tables": {}
            }
        }
        
        # Categorize tables (basic heuristic - you can improve this)
        for model in metadata.get("models", []):
            table_name = model.get("table_name", "").upper()
            
            # Basic categorization logic
            if any(keyword in table_name.lower() for keyword in ["trade", "order", "syndicate"]):
                domain = "trading_lifecycle"
            elif any(keyword in table_name.lower() for keyword in ["user", "auth", "login"]):
                domain = "user_management"
            elif any(keyword in table_name.lower() for keyword in ["tranche", "syndicate", "mars"]):
                domain = "syndicate_operations" 
            else:
                domain = "reporting"
            
            # Add table description (you might want to enhance this)
            table_desc = f"Table containing {table_name.lower().replace('_', ' ')} data"
            domains[domain]["tables"][table_name] = table_desc
        
        # Process views similarly
        for view in metadata.get("views", []):
            view_name = view.get("view_name", "").upper()
            
            # Basic categorization for views
            if any(keyword in view_name.lower() for keyword in ["trade", "order", "hedge"]):
                domain = "trading_lifecycle"
            elif any(keyword in view_name.lower() for keyword in ["user", "metrics"]):
                domain = "user_management"
            elif any(keyword in view_name.lower() for keyword in ["tranche", "syndicate", "termsheet"]):
                domain = "syndicate_operations"
            else:
                domain = "reporting"
            
            view_desc = f"View providing {view_name.lower().replace('_', ' ')} information"
            domains[domain]["tables"][view_name] = view_desc
        
        return domains
    
    def create_column_metadata_file(self, table_data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
        """Create column metadata for a single table."""
        
        column_metadata = {
            "table": table_name.upper(),
            "columns": {}
        }
        
        # Process columns
        for column in table_data.get("columns", []):
            col_name = column.get("name")
            if not col_name:
                continue
                
            col_info = {
                "type": column.get("type", "UNKNOWN"),
                "nullable": column.get("nullable", True),
                "description": self._generate_column_description(col_name, column),
            }
            
            # Add key information if present
            if column.get("key"):
                col_info["constraint"] = column["key"]
            
            # Add example values if present and useful
            example_values = column.get("example_values", [])
            if example_values and len(example_values) > 0:
                col_info["example_values"] = example_values[:5]  # Limit to 5 examples
            
            # Add join hints for foreign key-like columns
            if col_name.lower().endswith('_id') and col_name.lower() != table_name.lower() + '_id':
                col_info["join_hint"] = f"Likely references another table's ID field"
            
            column_metadata["columns"][col_name] = col_info
        
        return column_metadata
    
    def _generate_column_description(self, col_name: str, col_data: Dict[str, Any]) -> str:
        """Generate a basic description for a column based on its name and properties."""
        
        col_lower = col_name.lower()
        
        # Common patterns
        if col_lower.endswith('_id'):
            if col_lower == 'id' or col_lower.endswith('_id'):
                base_name = col_lower.replace('_id', '').replace('id', '')
                if base_name:
                    return f"Unique identifier for {base_name.replace('_', ' ')}"
                else:
                    return "Unique identifier"
        
        elif col_lower.endswith('_date') or col_lower.endswith('_time'):
            return f"Date/time when {col_lower.replace('_date', '').replace('_time', '').replace('_', ' ')} occurred"
        
        elif col_lower.endswith('_name'):
            return f"Name of the {col_lower.replace('_name', '').replace('_', ' ')}"
        
        elif col_lower.endswith('_status'):
            return f"Current status of the {col_lower.replace('_status', '').replace('_', ' ')}"
        
        elif col_lower in ['created_at', 'updated_at', 'modified_at']:
            return f"Timestamp when record was {col_lower.replace('_at', '').replace('_', ' ')}"
        
        elif col_lower in ['created_by', 'updated_by', 'modified_by']:
            return f"User who {col_lower.replace('_by', '').replace('_', ' ')} the record"
        
        else:
            # Default description
            return f"{col_name.replace('_', ' ').title()} field"
    
    def create_business_rules_template(self) -> Dict[str, Any]:
        """Create a template business rules file with common patterns."""
        
        return {
            "area": "date_handling",
            "description": "Rules for handling date and timestamp columns",
            "rules": [
                {
                    "pattern": "T00:00:00 timestamps",
                    "columns": ["*_date", "*_time"],
                    "rule": "Timestamps ending in T00:00:00 represent date-only values stored as timestamps",
                    "sql_guidance": "Use TRUNC() function for date comparisons",
                    "example": "WHERE TRUNC(trade_date) = DATE '2023-01-15'"
                },
                {
                    "pattern": "status_id columns", 
                    "columns": ["*_status_id", "status_id"],
                    "rule": "Status ID columns reference lookup tables for human-readable values",
                    "sql_guidance": "Join with appropriate lookup table or use CASE statements",
                    "example": "JOIN tranche_status_lookups tsl ON t.status_id = tsl.id"
                }
            ]
        }
    
    def restructure_all(self):
        """Main method to restructure all metadata."""
        
        logger.info("Starting metadata restructuring...")
        
        # Load existing metadata
        metadata = self.load_existing_metadata()
        
        # 1. Create business descriptions by domain
        logger.info("Creating business descriptions...")
        domains = self.create_business_descriptions(metadata)
        
        for domain_name, domain_data in domains.items():
            if domain_data["tables"]:  # Only create if there are tables
                output_file = self.output_dir.parent / "descriptions" / f"{domain_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(domain_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Created business description: {output_file}")
        
        # 2. Create individual column metadata files
        logger.info("Creating column metadata files...")
        
        # Process tables
        for model in metadata.get("models", []):
            table_name = model.get("table_name")
            if table_name:
                col_metadata = self.create_column_metadata_file(model, table_name)
                output_file = self.output_dir / f"{table_name.lower()}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(col_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Created column metadata: {output_file}")
        
        # Process views
        for view in metadata.get("views", []):
            view_name = view.get("view_name")
            if view_name:
                col_metadata = self.create_column_metadata_file(view, view_name)
                output_file = self.output_dir / f"{view_name.lower()}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(col_metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Created view metadata: {output_file}")
        
        # 3. Create business rules template
        logger.info("Creating business rules template...")
        business_rules = self.create_business_rules_template()
        rules_file = self.output_dir.parent / "business_rules" / "date_and_status_rules.json"
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(business_rules, f, indent=2, ensure_ascii=False)
        logger.info(f"Created business rules template: {rules_file}")
        
        logger.info("Metadata restructuring completed successfully!")


def main():
    """Main function to run metadata restructuring."""
    
    # Configuration
    INPUT_FILE = "meta_documents/p1-synd/schema/main_schema_metadata.json"
    OUTPUT_DIR = "meta_documents/p1-synd/columns"
    
    try:
        restructurer = ColumnMetadataRestructurer(INPUT_FILE, OUTPUT_DIR)
        restructurer.restructure_all()
        
        print("\nRestructuring completed!")
        print("Files created in:")
        print(f"  - Column metadata: {OUTPUT_DIR}")
        print(f"  - Business descriptions: meta_documents/p1-synd/descriptions/")
        print(f"  - Business rules: meta_documents/p1-synd/business_rules/")
        
    except Exception as e:
        logger.error(f"Restructuring failed: {e}")


if __name__ == "__main__":
    print("Column Metadata Restructuring Script")
    print("===================================")
    print("This script restructures your existing main_schema_metadata.json")
    print("into the new tiered metadata format with separate files for:")
    print("- Business descriptions by domain")
    print("- Individual table column metadata")
    print("- Business rules templates")
    print()
    
    # Ask for confirmation  
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled by user.")