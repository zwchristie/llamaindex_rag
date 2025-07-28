#!/usr/bin/env python3
"""Script to re-ingest schema documents using the new individual document approach."""

import json
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
from text_to_sql_rag.models.simple_models import DocumentType

def reingest_schema():
    """Re-ingest schema documents with new individual document approach."""
    
    # Initialize vector service
    vector_service = LlamaIndexVectorService()
    
    # Path to your actual schema file (adjust as needed)
    schema_path = "meta_documents/example_application/schema/main_schema_metadata.json"
    
    try:
        with open(schema_path, 'r') as f:
            schema_content = f.read()
        
        print("Loading schema content...")
        print(f"Schema file size: {len(schema_content)} characters")
        
        # Document ID for your schema
        document_id = "main_schema_metadata"
        
        # First, try to delete existing schema documents
        print("Attempting to delete existing schema documents...")
        try:
            vector_service.delete_document(document_id)
            print("✓ Deleted existing schema documents")
        except Exception as e:
            print(f"⚠ Could not delete existing documents (might not exist): {e}")
        
        # Add document using new individual document approach
        print("Adding schema using new individual document approach...")
        success = vector_service.add_document(
            document_id=document_id,
            content=schema_content,
            metadata={
                "source": schema_path,
                "document_name": "main_schema_metadata",
                "ingestion_method": "individual_documents"
            },
            document_type=DocumentType.SCHEMA.value
        )
        
        if success:
            print("✅ Successfully re-ingested schema with individual documents!")
            
            # Test retrieval
            print("\n🔍 Testing retrieval...")
            test_query = "What are all the deal names with tranches that are in announced status in Fixed Income?"
            results = vector_service.two_step_metadata_retrieval(
                query=test_query,
                similarity_top_k=5,
                document_type=DocumentType.SCHEMA.value
            )
            
            print(f"Test query: {test_query}")
            print(f"Results - Models: {len(results.get('models', []))}, Views: {len(results.get('views', []))}, Relationships: {len(results.get('relationships', []))}")
            
            # Show first model/view content
            models = results.get('models', [])
            views = results.get('views', [])
            
            if models:
                print(f"\n📋 First model content preview:")
                print(f"Model: {models[0]['metadata'].get('table_name')}")
                print(f"Content length: {len(models[0]['content'])}")
                print(f"Content preview: {models[0]['content'][:300]}...")
            
            if views:
                print(f"\n📋 First view content preview:")
                print(f"View: {views[0]['metadata'].get('view_name')}")
                print(f"Content length: {len(views[0]['content'])}")
                print(f"Content preview: {views[0]['content'][:300]}...")
                
        else:
            print("❌ Failed to re-ingest schema documents")
            
    except Exception as e:
        print(f"❌ Error during re-ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reingest_schema()