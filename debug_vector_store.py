#!/usr/bin/env python3
"""Debug script to see what's actually stored in the vector store."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
from text_to_sql_rag.models.simple_models import DocumentType

def debug_vector_store():
    """Debug what's actually stored in the vector store."""
    
    # Initialize vector service
    vector_service = LlamaIndexVectorService()
    
    print("🔍 DEBUGGING VECTOR STORE CONTENTS")
    print("=" * 60)
    
    # 1. List all documents
    print("\n1. ALL DOCUMENTS IN VECTOR STORE:")
    all_docs = vector_service.list_all_documents()
    print(f"Total documents: {len(all_docs)}")
    
    for i, doc in enumerate(all_docs[:10]):  # Show first 10
        print(f"  Doc {i+1}: {doc.get('document_id')} (type: {doc.get('document_type')}, chunks: {doc.get('chunk_count', 'unknown')})")
    
    # 2. Test search for schema documents
    print("\n2. SEARCHING FOR SCHEMA DOCUMENTS:")
    schema_results = vector_service.search_similar(
        query="database table schema",
        document_type=DocumentType.SCHEMA.value,
        similarity_top_k=10
    )
    
    print(f"Found {len(schema_results)} schema documents")
    for i, result in enumerate(schema_results[:5]):
        metadata = result.get('metadata', {})
        print(f"  Result {i+1}:")
        print(f"    ID: {result.get('id')}")
        print(f"    Document ID: {metadata.get('document_id')}")
        print(f"    Entity Type: {metadata.get('entity_type', 'unknown')}")
        print(f"    Table/View Name: {metadata.get('table_name') or metadata.get('view_name') or 'N/A'}")
        print(f"    Content Length: {len(result.get('content', ''))}")
        print(f"    Content Preview: {result.get('content', '')[:150]}...")
        print(f"    Metadata Keys: {list(metadata.keys())}")
        print()
    
    # 3. Test the specific query
    print("\n3. TESTING SPECIFIC FIXED INCOME QUERY:")
    test_query = "What are all the deal names with tranches that are in announced status in Fixed Income?"
    print(f"Query: {test_query}")
    
    # Test regular search
    print("\n3a. Regular search results:")
    regular_results = vector_service.search_similar(
        query=test_query,
        document_type=DocumentType.SCHEMA.value,
        similarity_top_k=5
    )
    
    for i, result in enumerate(regular_results):
        metadata = result.get('metadata', {})
        print(f"  Regular Result {i+1}:")
        print(f"    Entity: {metadata.get('table_name') or metadata.get('view_name') or metadata.get('relationship_name', 'Unknown')}")
        print(f"    Type: {metadata.get('entity_type', 'unknown')}")
        print(f"    Score: {result.get('combined_score', result.get('score', 0.0)):.3f}")
        print(f"    Content Length: {len(result.get('content', ''))}")
    
    # Test 2-step retrieval
    print("\n3b. 2-Step retrieval results:")
    two_step_results = vector_service.two_step_metadata_retrieval(
        query=test_query,
        similarity_top_k=5,
        document_type=DocumentType.SCHEMA.value
    )
    
    models = two_step_results.get('models', [])
    views = two_step_results.get('views', [])
    relationships = two_step_results.get('relationships', [])
    
    print(f"  2-Step Results: {len(models)} models, {len(views)} views, {len(relationships)} relationships")
    
    for i, model in enumerate(models):
        metadata = model.get('metadata', {})
        print(f"    Model {i+1}: {metadata.get('table_name')} (content: {len(model.get('content', ''))} chars)")
    
    for i, view in enumerate(views):
        metadata = view.get('metadata', {})
        print(f"    View {i+1}: {metadata.get('view_name')} (content: {len(view.get('content', ''))} chars)")
    
    for i, rel in enumerate(relationships):
        metadata = rel.get('metadata', {})
        print(f"    Relationship {i+1}: {metadata.get('relationship_name')} (content: {len(rel.get('content', ''))} chars)")
    
    # 4. Show one complete document if found
    if schema_results:
        print(f"\n4. COMPLETE CONTENT OF FIRST SCHEMA DOCUMENT:")
        print("=" * 60)
        first_result = schema_results[0]
        print(f"Document ID: {first_result.get('id')}")
        print(f"Full Content:\n{first_result.get('content', '')}")
        print("=" * 60)

if __name__ == "__main__":
    debug_vector_store()