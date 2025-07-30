#!/usr/bin/env python3
"""Test script to verify lookup document processing works correctly."""

import json
import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from text_to_sql_rag.utils.content_processor import ContentProcessor
from text_to_sql_rag.models.simple_models import DocumentType

def test_lookup_document_processing():
    """Test lookup document processing to debug sync issues."""
    
    # Sample tranche status lookup data (similar to what might be in tranche_status_lookups.json)
    lookup_data = {
        "name": "tranche_status",
        "description": "Status values for tranche records in the database",
        "values": [
            {"id": 1, "name": "Active", "code": "ACT"},
            {"id": 2, "name": "Inactive", "code": "INACT"},
            {"id": 3, "name": "Pending", "code": "PEND"},
            {"id": 4, "name": "Cancelled", "code": "CANC"},
            {"id": 5, "name": "Completed", "code": "COMP"}
        ]
    }
    
    print("=== TESTING LOOKUP DOCUMENT PROCESSING ===")
    print(f"Original lookup data keys: {list(lookup_data.keys())}")
    print(f"Number of values: {len(lookup_data['values'])}")
    
    # Test the document creation
    processor = ContentProcessor()
    json_content = json.dumps(lookup_data)
    
    print(f"JSON content length: {len(json_content)} characters")
    print(f"Is JSON content: {processor.is_json_content(json_content)}")
    
    # Create individual documents for lookup
    try:
        documents = processor.create_individual_documents(json_content, DocumentType.LOOKUP_METADATA)
        print(f"Number of documents created: {len(documents)}")
        
        if documents:
            lookup_doc = documents[0]
            print("\n=== LOOKUP DOCUMENT DETAILS ===")
            print(f"Content length: {len(lookup_doc['content'])} characters")
            print(f"Metadata keys: {list(lookup_doc['metadata'].keys())}")
            
            metadata = lookup_doc['metadata']
            print(f"Chunk type: {metadata.get('chunk_type')}")
            print(f"Entity type: {metadata.get('entity_type')}")
            print(f"Lookup name: {metadata.get('lookup_name')}")
            print(f"Value count: {metadata.get('value_count')}")
            print(f"Search terms count: {len(metadata.get('search_terms', []))}")
            
            print("\n=== CONTENT PREVIEW ===")
            content_lines = lookup_doc['content'].split('\n')
            print(f"Total lines: {len(content_lines)}")
            print("First 15 lines:")
            for i, line in enumerate(content_lines[:15]):
                print(f"{i+1:2d}: {line}")
            
            print("\n=== VALIDATION CHECKS ===")
            content = lookup_doc['content']
            if len(content) > 10:
                print("[PASS] Content length is sufficient")
            else:
                print(f"[FAIL] Content too short: {len(content)} characters")
            
            if "tranche_status" in content.lower():
                print("[PASS] Lookup name found in content")
            else:
                print("[FAIL] Lookup name not found in content")
                
            if "Active" in content:
                print("[PASS] Lookup values found in content")
            else:
                print("[FAIL] Lookup values not found in content")
                
            if metadata.get('entity_type') == 'lookup':
                print("[PASS] Entity type is 'lookup'")
            else:
                print(f"[FAIL] Entity type is '{metadata.get('entity_type')}', expected 'lookup'")
                
            if metadata.get('lookup_name') == 'tranche_status':
                print("[PASS] Lookup name in metadata is correct")
            else:
                print(f"[FAIL] Lookup name in metadata is '{metadata.get('lookup_name')}', expected 'tranche_status'")
            
            return True
        else:
            print("[FAIL] No documents created!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception during document creation: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_lookup_processing():
    """Test fallback lookup processing for unknown structures."""
    
    print("\n=== TESTING FALLBACK LOOKUP PROCESSING ===")
    
    # Test with unstructured lookup data
    unknown_lookup = {
        "some_key": "some_value",
        "another_key": {"data": ["item1", "item2", "item3"]}
    }
    
    processor = ContentProcessor()
    json_content = json.dumps(unknown_lookup)
    
    try:
        documents = processor.create_individual_documents(json_content, DocumentType.LOOKUP_METADATA)
        print(f"Fallback documents created: {len(documents)}")
        
        if documents:
            doc = documents[0]
            print(f"Fallback content length: {len(doc['content'])}")
            print(f"Fallback chunk type: {doc['metadata'].get('chunk_type')}")
            print(f"Fallback entity type: {doc['metadata'].get('entity_type')}")
            
            if doc['metadata'].get('chunk_type') == 'lookup_fallback':
                print("[PASS] Fallback processing works")
            else:
                print("[FAIL] Fallback processing failed")
            
            return True
        else:
            print("[FAIL] No fallback documents created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Fallback processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_lookup_document_processing()
    success2 = test_fallback_lookup_processing()
    
    if success1 and success2:
        print("\n[SUCCESS] All lookup document processing tests passed!")
    else:
        print("\n[FAIL] Some tests failed!")
        sys.exit(1)