#!/usr/bin/env python3
"""Test script to verify the USERS document creation fixes."""

import json
import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from text_to_sql_rag.utils.content_processor import ContentProcessor
from text_to_sql_rag.models.simple_models import DocumentType

def test_users_document_creation():
    """Test creating a document from the USERS JSON model."""
    
    # Sample USERS data based on your original structure
    users_model = {
        "name": "USERS",
        "properties": {
            "description": "Contains information about the user based on their sid being the username, containing information about their name, role, and team"
        },
        "columns": [
            {"name": "team_id", "type": "VARCHAR"},
            {"name": "username", "type": "VARCHAR", "notNull": True},
            {"name": "sid", "type": "VARCHAR"},
            {"name": "company", "type": "VARCHAR"},
            {"name": "first_name", "type": "VARCHAR"},
            {"name": "last_name", "type": "VARCHAR"},
            {"name": "email", "type": "VARCHAR"},
            {"name": "product_ids", "type": "VARCHAR"},
            {"name": "status_ids", "type": "VARCHAR"},
            {"name": "deal_ids", "type": "VARCHAR"},
            {"name": "tranche_ids", "type": "VARCHAR"},
            {"name": "created_by", "type": "VARCHAR", "notNull": True},
            {
                "name": "created_date", 
                "type": "DATE", 
                "example_values": ["2012-01-07T18:11:00", "2016-03-13T23:49:39", "2016-03-27T03:48:53"], 
                "notNull": True
            },
            {"name": "updated_by", "type": "VARCHAR", "notNull": True},
            {
                "name": "updated_date", 
                "type": "DATE", 
                "example_values": ["2024-04-19T17:45:36", "2025-04-01T03:43:56", "2025-04-03T10:51:09", "2025-04-03T10:51:14", "2025-04-22T18:00:52"], 
                "notNull": True
            },
            {"name": "is_deleted", "type": "NUMERIC", "example_values": [0, 1], "default": "0"},
            {"name": "region", "type": "NUMERIC", "example_values": [1, 2, 9]},
            {"name": "salesone_id", "type": "NUMERIC", "example_values": [1, 2, 3, 4, 5]},
            {
                "name": "ps_last_login", 
                "type": "DATE", 
                "example_values": ["2022-03-17T03:29:47", "2025-02-28T08:19:39", "2025-03-05T05:43:31", "2025-03-06T09:50:20", "2025-03-07T11:20:00"]
            },
            {
                "name": "p1_last_login", 
                "type": "DATE", 
                "example_values": ["2020-09-09T04:24:03", "2023-04-26T04:07:49", "2024-05-22T03:06:14", "2024-09-17T10:20:41", "2024-09-18T01:11:10"]
            },
            {"name": "primary_team_id", "type": "NUMERIC", "example_values": [927, 936, 3066, 3120, 3596]},
            {"name": "timezone", "type": "VARCHAR"},
            {"name": "role_override", "type": "VARCHAR"},
            {"name": "super_user", "type": "NUMERIC", "example_values": [0], "default": "0"},
            {
                "name": "roadshow_sync_date", 
                "type": "DATE", 
                "example_values": ["2024-04-19T17:45:36", "2025-04-01T03:43:56", "2025-04-03T10:51:09", "2025-04-03T10:51:14", "2025-04-22T18:00:52"]
            },
            {"name": "is_coordinator", "type": "NUMERIC", "example_values": [0]},
            {
                "name": "p1_loans_last_login", 
                "type": "DATE", 
                "example_values": ["2023-02-24T15:49:10", "2024-01-09T09:02:38", "2025-02-28T12:23:14", "2025-03-04T10:50:10", "2025-06-12T10:06:22"]
            },
            {"name": "team_email", "type": "VARCHAR"},
            {"name": "manager_name", "type": "VARCHAR"}
        ],
        "primaryKey": "USERNAME",
        "refSql": "SELECT * FROM SYND.USERS"
    }
    
    # Create schema metadata structure
    schema_metadata = {
        "catalog": "fixed_income_oracle_sql_schema",
        "schema": "synd",
        "models": [users_model]
    }
    
    # Test the document creation
    processor = ContentProcessor()
    json_content = json.dumps(schema_metadata)
    
    print("=== TESTING USERS DOCUMENT CREATION ===")
    print(f"Original JSON length: {len(json_content)} characters")
    
    # Create individual documents
    documents = processor.create_individual_documents(json_content, DocumentType.SCHEMA)
    
    print(f"Number of documents created: {len(documents)}")
    
    # Find the USERS document
    users_doc = None
    for doc in documents:
        if doc["metadata"].get("table_name") == "USERS":
            users_doc = doc
            break
    
    if users_doc:
        print("\n=== USERS DOCUMENT FOUND ===")
        print(f"Content length: {len(users_doc['content'])} characters")
        print(f"Metadata keys: {list(users_doc['metadata'].keys())}")
        print(f"Table name: {users_doc['metadata'].get('table_name')}")
        print(f"Entity type: {users_doc['metadata'].get('entity_type')}")
        print(f"Classified entity type: {users_doc['metadata'].get('classified_entity_type')}")
        print(f"Number of columns: {len(users_doc['metadata'].get('columns', []))}")
        print(f"Primary key: {users_doc['metadata'].get('primary_key')}")
        print(f"Business terms count: {len(users_doc['metadata'].get('business_terms', []))}")
        print(f"Searchable content length: {len(users_doc['metadata'].get('searchable_content', ''))}")
        
        print("\n=== BUSINESS TERMS ===")
        business_terms = users_doc['metadata'].get('business_terms', [])
        print(f"Business terms: {business_terms[:10]}...")  # Show first 10
        
        print("\n=== SEARCHABLE CONTENT PREVIEW ===")
        searchable_content = users_doc['metadata'].get('searchable_content', '')
        print(f"First 200 chars: {searchable_content[:200]}...")
        
        print("\n=== DOCUMENT CONTENT PREVIEW ===")
        content = users_doc['content']
        lines = content.split('\n')
        print(f"Total lines: {len(lines)}")
        print("First 20 lines:")
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:2d}: {line}")
        
        print("\n=== COLUMN INFORMATION CHECK ===")
        # Look for specific column information in content
        if "team_id" in content:
            print("[PASS] team_id column found in content")
        if "VARCHAR" in content:
            print("[PASS] VARCHAR data type found in content")
        if "example_values" in content.lower() or "example values" in content.lower():
            print("[PASS] Example values found in content")
        if "2024-04-19" in content:
            print("[PASS] Specific example values found in content")
        if "primary key" in content.lower():
            print("[PASS] Primary key information found in content")
        if "username" in content.lower():
            print("[PASS] Primary key value found in content")
        
        return True
    else:
        print("[FAIL] USERS document not found!")
        print("Available documents:")
        for doc in documents:
            print(f"  - {doc['metadata'].get('table_name', 'Unknown')}")
        return False

if __name__ == "__main__":
    success = test_users_document_creation()
    if success:
        print("\n[SUCCESS] Test completed successfully!")
    else:
        print("\n[FAIL] Test failed!")
        sys.exit(1)