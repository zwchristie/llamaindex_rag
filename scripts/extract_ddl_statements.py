#!/usr/bin/env python3
"""
Housekeeping script to extract DDL statements from database and create individual .sql files.
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

try:
    from sqlalchemy import create_engine, MetaData, Table, inspect
    from sqlalchemy.schema import CreateTable
    from sqlalchemy.dialects import oracle
except ImportError:
    logger.error("SQLAlchemy not installed. Please run: pip install sqlalchemy cx_Oracle")
    exit(1)


class DDLExtractor:
    """Extract DDL statements for tables/views and save to individual files."""
    
    def __init__(self, connection_string: str, output_dir: str):
        """
        Initialize DDL extractor.
        
        Args:
            connection_string: Oracle database connection string
            output_dir: Directory to save DDL files
        """
        self.connection_string = connection_string
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        try:
            self.engine = create_engine(connection_string)
            self.metadata = MetaData()
            self.inspector = inspect(self.engine)
            logger.info(f"Connected to database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_table_descriptions(self, descriptions_file: str) -> Dict[str, str]:
        """Load table descriptions from fi_table_details_demo.json file."""
        try:
            with open(descriptions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract descriptions - assuming format: {table_name: {description: "..."}}
            descriptions = {}
            for table_name, details in data.items():
                if isinstance(details, dict) and 'description' in details:
                    descriptions[table_name.upper()] = details['description']
                else:
                    logger.warning(f"No description found for table: {table_name}")
                    descriptions[table_name.upper()] = f"No description available for {table_name}"
            
            logger.info(f"Loaded descriptions for {len(descriptions)} tables/views")
            return descriptions
            
        except FileNotFoundError:
            logger.error(f"Description file not found: {descriptions_file}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load descriptions: {e}")
            return {}
    
    def get_table_ddl(self, table_name: str, schema: str = None) -> str:
        """Extract DDL for a specific table."""
        try:
            # Reflect the table
            table = Table(table_name, self.metadata, autoload_with=self.engine, schema=schema)
            
            # Generate CREATE TABLE statement
            create_statement = CreateTable(table).compile(dialect=oracle.dialect())
            
            return str(create_statement)
            
        except Exception as e:
            logger.error(f"Failed to get DDL for table {table_name}: {e}")
            return None
    
    def get_view_ddl(self, view_name: str, schema: str = None) -> str:
        """Extract DDL for a specific view (this is more complex for views)."""
        try:
            # For views, we'll create a simplified structure since getting the actual
            # CREATE VIEW statement requires additional privileges
            view_columns = self.inspector.get_columns(view_name, schema=schema)
            
            ddl_lines = [f"-- {view_name} view structure"]
            ddl_lines.append(f"-- Note: This is a simplified view structure")
            ddl_lines.append(f"CREATE OR REPLACE VIEW {view_name} AS")
            ddl_lines.append("SELECT")
            
            column_lines = []
            for col in view_columns:
                col_type = str(col['type'])
                nullable = "" if col.get('nullable', True) else " NOT NULL"
                column_lines.append(f"    {col['name']} {col_type}{nullable}")
            
            ddl_lines.append(",\n".join(column_lines))
            ddl_lines.append("FROM <source_tables>;")
            
            return "\n".join(ddl_lines)
            
        except Exception as e:
            logger.error(f"Failed to get DDL for view {view_name}: {e}")
            return None
    
    def extract_all_ddl(self, table_names: List[str], descriptions: Dict[str, str], schema: str = None):
        """Extract DDL for all specified tables/views."""
        logger.info(f"Starting DDL extraction for {len(table_names)} objects")
        
        success_count = 0
        error_count = 0
        
        for table_name in table_names:
            try:
                table_name_upper = table_name.upper()
                
                # Check if it's a table or view
                tables = self.inspector.get_table_names(schema=schema)
                views = self.inspector.get_view_names(schema=schema)
                
                # Get description
                description = descriptions.get(table_name_upper, f"No description available for {table_name}")
                
                if table_name_upper in [t.upper() for t in tables]:
                    # It's a table
                    ddl = self.get_table_ddl(table_name, schema)
                    object_type = "table"
                elif table_name_upper in [v.upper() for v in views]:
                    # It's a view
                    ddl = self.get_view_ddl(table_name, schema)
                    object_type = "view"
                else:
                    logger.warning(f"Object {table_name} not found in database")
                    continue
                
                if ddl:
                    # Add description as comment at the top
                    full_ddl = f"-- {table_name} {object_type} - {description}\n{ddl}"
                    
                    # Save to file
                    output_file = self.output_dir / f"{table_name.lower()}.sql"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(full_ddl)
                    
                    logger.info(f"Saved DDL for {object_type} {table_name} to {output_file}")
                    success_count += 1
                else:
                    logger.error(f"Failed to generate DDL for {table_name}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {table_name}: {e}")
                error_count += 1
        
        logger.info(f"DDL extraction completed. Success: {success_count}, Errors: {error_count}")


def main():
    """Main function to run DDL extraction."""
    
    # Configuration - UPDATE THESE VALUES
    DATABASE_URL = "oracle://username:password@host:port/service_name"
    DESCRIPTIONS_FILE = "meta_documents/p1-synd/fi_table_details_demo.json"
    OUTPUT_DIR = "meta_documents/p1-synd/schema/ddl"
    SCHEMA_NAME = None  # Set to your schema name if needed
    
    # Check if descriptions file exists
    if not os.path.exists(DESCRIPTIONS_FILE):
        logger.error(f"Please place your fi_table_details_demo.json file at: {DESCRIPTIONS_FILE}")
        logger.info("The file should contain table names as keys with description objects as values")
        return
    
    try:
        # Initialize extractor
        extractor = DDLExtractor(DATABASE_URL, OUTPUT_DIR)
        
        # Load table descriptions
        descriptions = extractor.load_table_descriptions(DESCRIPTIONS_FILE)
        
        if not descriptions:
            logger.error("No table descriptions loaded. Cannot proceed.")
            return
        
        # Get list of table names from descriptions
        table_names = list(descriptions.keys())
        
        # Extract DDL for all tables
        extractor.extract_all_ddl(table_names, descriptions, SCHEMA_NAME)
        
        logger.info("DDL extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"DDL extraction failed: {e}")


if __name__ == "__main__":
    print("DDL Extraction Script")
    print("====================")
    print("This script extracts DDL statements from your Oracle database")
    print("and creates individual .sql files for each table/view.")
    print()
    print("Before running, please:")
    print("1. Install SQLAlchemy: pip install sqlalchemy cx_Oracle")
    print("2. Update the DATABASE_URL in the script")
    print("3. Place fi_table_details_demo.json in the correct location")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ")
    if response.lower() == 'y':
        main()
    else:
        print("Cancelled by user.")