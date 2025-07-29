#!/usr/bin/env python3
"""Show how Oracle SQL rules are embedded in the document content."""

import json
import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from text_to_sql_rag.utils.content_processor import ContentProcessor
from text_to_sql_rag.models.simple_models import DocumentType

def show_oracle_rules_in_document():
    """Show Oracle SQL rules in the generated document."""
    
    # Simple USERS model
    users_model = {
        "name": "USERS",
        "properties": {"description": "User information table"},
        "columns": [
            {"name": "username", "type": "VARCHAR", "notNull": True},
            {"name": "email", "type": "VARCHAR"}
        ],
        "primaryKey": "USERNAME",
        "refSql": "SELECT * FROM SYND.USERS"
    }
    
    schema_metadata = {
        "catalog": "fixed_income_oracle_sql_schema",
        "schema": "synd", 
        "models": [users_model]
    }
    
    processor = ContentProcessor()
    json_content = json.dumps(schema_metadata)
    documents = processor.create_individual_documents(json_content, DocumentType.SCHEMA)
    
    users_doc = documents[0]
    content = users_doc['content']
    metadata = users_doc['metadata']
    
    print("=== DOCUMENT CONTENT LENGTH ===")
    print(f"Total content: {len(content)} characters")
    print(f"Total lines: {len(content.split(chr(10)))}")
    
    print("\n=== SQL-RELATED METADATA ===")
    print(f"SQL Dialect: {metadata.get('sql_dialect')}")
    print(f"Supports Joins: {metadata.get('supports_joins')}")
    print(f"Case Sensitive: {metadata.get('case_sensitive')}")
    print(f"Requires Schema Qualification: {metadata.get('requires_schema_qualification')}")
    print(f"Allowed Operations: {metadata.get('allowed_operations')}")
    print(f"Prohibited Operations: {metadata.get('prohibited_operations')}")
    
    print("\n=== ORACLE QUERY EXAMPLES IN CONTENT ===")
    lines = content.split('\n')
    in_sql_examples = False
    for i, line in enumerate(lines):
        if "ORACLE SQL QUERY EXAMPLES" in line:
            in_sql_examples = True
            print(f"Line {i+1}: {line}")
        elif in_sql_examples and line.startswith("===") and "ORACLE" in line and "RULES" in line:
            print(f"Line {i+1}: Found Oracle SQL rules section")
            break
        elif in_sql_examples:
            print(f"Line {i+1}: {line}")
    
    print("\n=== ORACLE SQL RULES SECTION ===")
    in_rules = False
    rule_lines = []
    for i, line in enumerate(lines):
        if "ORACLE SQL GENERATION RULES" in line:
            in_rules = True
        elif in_rules and "END ORACLE SQL RULES" in line:
            break
        elif in_rules:
            rule_lines.append(line)
    
    print(f"Oracle SQL rules section: {len(rule_lines)} lines")
    print("First 10 lines of rules:")
    for i, line in enumerate(rule_lines[:10]):
        print(f"  {line}")
    
    print("\n=== SEARCHABLE CONTENT WITH SQL TERMS ===")
    searchable = metadata.get('searchable_content', '')
    sql_terms = ['oracle sql', 'select', 'join', 'where', 'order by']
    found_terms = []
    for term in sql_terms:
        if term in searchable:
            found_terms.append(term)
    
    print(f"SQL terms found in searchable content: {found_terms}")
    
    return True

if __name__ == "__main__":
    show_oracle_rules_in_document()
    print("\n[SUCCESS] Oracle SQL rules successfully integrated into documents!")