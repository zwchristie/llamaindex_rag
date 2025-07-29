#!/usr/bin/env python3
"""
Simple test for semantic chunking without requiring full environment setup.
"""

import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from text_to_sql_rag.utils.content_processor import ContentProcessor
from text_to_sql_rag.models.simple_models import DocumentType


def test_schema_parsing():
    """Test schema parsing with different query scenarios."""
    print("Testing Schema Parsing for SQL Generation Scenarios")
    print("=" * 80)
    
    # Load the example schema
    schema_path = Path("meta_documents/example_application/schema/main_schema_metadata.json")
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return
    
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    # Test semantic chunking
    processor = ContentProcessor()
    chunks = processor.create_semantic_chunks(schema_content, DocumentType.SCHEMA)
    
    print(f"Created {len(chunks)} semantic entities")
    print()
    
    # Analyze what we have for SQL generation
    entities_by_type = {}
    for chunk in chunks:
        chunk_type = chunk['metadata']['chunk_type']
        if chunk_type not in entities_by_type:
            entities_by_type[chunk_type] = []
        entities_by_type[chunk_type].append(chunk)
    
    print("Available Entities by Type:")
    for entity_type, entity_list in entities_by_type.items():
        print(f"  {entity_type}: {len(entity_list)} entities")
    print()
    
    # Test common SQL query scenarios
    scenarios = [
        {
            "query": "Show me all users",
            "expected_entities": ["users table"],
            "test_keywords": ["users", "user", "table"]
        },
        {
            "query": "Get user email and status",
            "expected_entities": ["users table with email and status columns"],
            "test_keywords": ["users", "email", "status", "column"]
        },
        {
            "query": "Users and their orders",
            "expected_entities": ["users table", "orders table", "user_orders relationship"],
            "test_keywords": ["users", "orders", "relationship", "join"]
        },
        {
            "query": "Total orders per user",
            "expected_entities": ["user_order_summary view or users/orders for aggregation"],
            "test_keywords": ["user", "orders", "total", "summary", "count"]
        },
        {
            "query": "Order items with product details",
            "expected_entities": ["order_items table", "orders relationship"],
            "test_keywords": ["order_items", "product", "items", "relationship"]
        }
    ]
    
    print("Testing SQL Query Scenarios")
    print("-" * 60)
    
    for scenario in scenarios:
        print(f"\nScenario: '{scenario['query']}'")
        print(f"Expected: {scenario['expected_entities']}")
        print("Matching entities:")
        
        matches = []
        for chunk in chunks:
            content = chunk['content'].lower()
            metadata = chunk['metadata']
            
            # Score the chunk based on keyword matches
            score = 0
            for keyword in scenario['test_keywords']:
                if keyword.lower() in content:
                    score += 1
            
            if score > 0:
                entity_name = metadata.get('table_name', metadata.get('view_name', metadata.get('business_domain', 'unknown')))
                matches.append({
                    'entity': entity_name,
                    'type': metadata['chunk_type'],
                    'score': score
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        if matches:
            for match in matches[:3]:  # Top 3 matches
                print(f"  MATCH {match['type']}: {match['entity']} (score: {match['score']})")
        else:
            print("  NO MATCH - No matching entities found - potential SQL generation issue")
    
    print("\n" + "=" * 80)
    print("Detailed Entity Analysis")
    print("=" * 80)
    
    # Show each entity type in detail
    for entity_type, entity_list in entities_by_type.items():
        print(f"\n--- {entity_type.upper()} ENTITIES ---")
        
        for i, chunk in enumerate(entity_list):
            metadata = chunk['metadata']
            entity_name = metadata.get('table_name', metadata.get('view_name', metadata.get('business_domain', f'entity_{i}')))
            
            print(f"\nEntity: {entity_name}")
            
            # Show key metadata
            if 'columns' in metadata:
                print(f"  Columns: {', '.join(metadata['columns'][:5])}" + ("..." if len(metadata['columns']) > 5 else ""))
            
            if 'business_terms' in metadata:
                print(f"  Business terms: {', '.join(metadata['business_terms'][:5])}")
            
            if 'related_tables' in metadata:
                print(f"  Related tables: {', '.join(metadata['related_tables'])}")
            
            # Show content preview
            content_preview = chunk['content'][:200].replace('\n', ' ')
            print(f"  Content: {content_preview}...")
    
    print("\n" + "=" * 80)
    print("SQL Generation Readiness Assessment")
    print("=" * 80)
    
    # Check if we have everything needed for good SQL generation
    has_tables = any(chunk['metadata']['chunk_type'] == 'table_entity' for chunk in chunks)
    has_relationships = any(chunk['metadata']['chunk_type'] == 'relationship_domain' for chunk in chunks)
    has_views = any(chunk['metadata']['chunk_type'] == 'view_entity' for chunk in chunks)
    has_overview = any(chunk['metadata']['chunk_type'] == 'schema_overview' for chunk in chunks)
    
    print(f"Tables available: {has_tables}")
    print(f"Relationships available: {has_relationships}")
    print(f"Views available: {has_views}")
    print(f"Schema overview available: {has_overview}")
    
    # Count detailed information
    total_columns = sum(len(chunk['metadata'].get('columns', [])) for chunk in chunks)
    total_business_terms = sum(len(chunk['metadata'].get('business_terms', [])) for chunk in chunks)
    
    print(f"Total columns documented: {total_columns}")
    print(f"Total business terms: {total_business_terms}")
    
    if all([has_tables, has_relationships]):
        print("\nSUCCESS: Schema is well-structured for SQL generation!")
        print("   - Individual table entities with complete column info")
        print("   - Relationship information for joins")
        print("   - Business context for better understanding")
    else:
        print("\nWARNING: Schema may need improvements for optimal SQL generation")


if __name__ == "__main__":
    test_schema_parsing()