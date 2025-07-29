#!/usr/bin/env python3
"""Test script to verify individual document creation is working correctly."""

import json
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_to_sql_rag.utils.content_processor import ContentProcessor
from text_to_sql_rag.models.simple_models import DocumentType

def test_document_creation():
    """Test that individual documents are created correctly."""
    # Load the sample schema
    schema_path = "meta_documents/example_application/schema/main_schema_metadata.json"
    
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    # Create content processor
    processor = ContentProcessor()
    
    # Create individual documents
    individual_docs = processor.create_individual_documents(schema_content, DocumentType.SCHEMA)
    
    print(f"Created {len(individual_docs)} individual documents:")
    print("=" * 80)
    
    for i, doc in enumerate(individual_docs):
        metadata = doc["metadata"]
        entity_type = metadata.get("entity_type", "unknown")
        entity_name = (metadata.get("table_name") or 
                      metadata.get("view_name") or 
                      metadata.get("relationship_name") or "unknown")
        
        print(f"\nDocument {i+1}: {entity_type.upper()} - {entity_name}")
        print("-" * 40)
        print("Content length:", len(doc["content"]))
        print("First 200 chars:", doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"])
        print("Metadata keys:", list(metadata.keys()))
        
        # Show full content for one example
        if i == 0:
            print("\nFULL CONTENT OF FIRST DOCUMENT:")
            print("=" * 80)
            print(doc["content"])
            print("=" * 80)

if __name__ == "__main__":
    test_document_creation()