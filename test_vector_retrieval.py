#!/usr/bin/env python3
"""
Local test file for vector store retrieval testing.
Use this to quickly test search queries and see where the search is not hitting.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
from text_to_sql_rag.services.bedrock_service import BedrockEmbeddingService, BedrockLLMService
from text_to_sql_rag.config.settings import settings


class VectorStoreTestRunner:
    """Test runner for vector store retrieval testing."""
    
    def __init__(self):
        """Initialize test runner with vector service."""
        print("🔧 Initializing Vector Store Test Runner...")
        
        # Initialize services
        self.bedrock_embedding = BedrockEmbeddingService()
        self.bedrock_llm = BedrockLLMService()
        
        # Initialize vector service
        self.vector_service = LlamaIndexVectorService(
            bedrock_embedding=self.bedrock_embedding,
            bedrock_llm=self.bedrock_llm
        )
        
        print("✅ Vector Store Test Runner initialized")
    
    def test_search_query(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Test a search query and return results."""
        print(f"\n🔍 Testing search query: '{query}'")
        print("-" * 60)
        
        try:
            results = self.vector_service.search_similar(
                query=query,
                max_results=max_results,
                document_type="schema"
            )
            
            print(f"📊 Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"ID: {result.get('id', 'N/A')}")
                print(f"Score: {result.get('score', 'N/A'):.4f}")
                
                # Show metadata
                metadata = result.get('metadata', {})
                chunk_type = metadata.get('chunk_type', 'N/A')
                table_name = metadata.get('table_name', metadata.get('view_name', 'N/A'))
                print(f"Type: {chunk_type}")
                if table_name != 'N/A':
                    print(f"Entity: {table_name}")
                
                # Show content preview
                content = result.get('content', '')
                print(f"Content preview:")
                print(content[:300] + "..." if len(content) > 300 else content)
                print()
            
            return results
            
        except Exception as e:
            print(f"❌ Error during search: {str(e)}")
            return []
    
    def test_multiple_queries(self, queries: List[str]) -> None:
        """Test multiple queries and compare results."""
        print("\n🧪 Testing Multiple Queries")
        print("=" * 80)
        
        all_results = {}
        
        for query in queries:
            results = self.test_search_query(query, max_results=3)
            all_results[query] = results
        
        # Summary comparison
        print("\n📋 SUMMARY COMPARISON")
        print("=" * 80)
        
        for query, results in all_results.items():
            print(f"\nQuery: '{query}' -> {len(results)} results")
            if results:
                top_result = results[0]
                metadata = top_result.get('metadata', {})
                chunk_type = metadata.get('chunk_type', 'N/A')
                entity_name = metadata.get('table_name', metadata.get('view_name', 'N/A'))
                print(f"  Top result: {chunk_type} - {entity_name} (score: {top_result.get('score', 0):.4f})")
            else:
                print("  No results found!")
    
    def analyze_schema_coverage(self) -> None:
        """Analyze what schema entities are available in the vector store."""
        print("\n🔍 Analyzing Schema Coverage")
        print("=" * 80)
        
        # Test broad schema queries to see what's available
        broad_queries = [
            "database tables",
            "schema information", 
            "table columns",
            "views",
            "relationships"
        ]
        
        for query in broad_queries:
            print(f"\n--- Query: '{query}' ---")
            results = self.vector_service.search_similar(
                query=query,
                max_results=10,
                document_type="schema"
            )
            
            if results:
                entity_types = {}
                for result in results:
                    metadata = result.get('metadata', {})
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    entity_name = metadata.get('table_name', metadata.get('view_name', 'unnamed'))
                    
                    if chunk_type not in entity_types:
                        entity_types[chunk_type] = []
                    entity_types[chunk_type].append(entity_name)
                
                for entity_type, entities in entity_types.items():
                    print(f"  {entity_type}: {', '.join(set(entities))}")
            else:
                print(f"  No results found for '{query}'")
    
    def test_sql_generation_scenarios(self) -> None:
        """Test common SQL generation scenarios."""
        print("\n💡 Testing SQL Generation Scenarios")
        print("=" * 80)
        
        scenarios = [
            # Basic table queries
            {
                "query": "show me all users",
                "expected": "Should find users table"
            },
            {
                "query": "get user information with email",
                "expected": "Should find users table with email column"
            },
            # Join scenarios
            {
                "query": "users and their orders",
                "expected": "Should find users/orders relationship"
            },
            {
                "query": "order details with items",
                "expected": "Should find orders/order_items relationship"  
            },
            # Aggregation scenarios
            {
                "query": "total orders per user",
                "expected": "Should find user_order_summary view or suggest aggregation"
            },
            {
                "query": "count of orders by status",
                "expected": "Should find orders table with status column"
            },
            # Column-specific queries
            {
                "query": "user status information",
                "expected": "Should find users table status column"
            },
            {
                "query": "order dates and amounts",
                "expected": "Should find orders table with date/amount columns"
            }
        ]
        
        for scenario in scenarios:
            print(f"\n--- Scenario: '{scenario['query']}' ---")
            print(f"Expected: {scenario['expected']}")
            
            results = self.test_search_query(scenario['query'], max_results=2)
            
            if results:
                print("✅ Results found - analyze if they match expectation")
            else:
                print("❌ No results - this query might fail in SQL generation")
    
    def run_interactive_mode(self) -> None:
        """Run interactive testing mode."""
        print("\n🎮 Interactive Mode - Enter queries to test (type 'quit' to exit)")
        print("-" * 60)
        
        while True:
            try:
                query = input("\nEnter search query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                self.test_search_query(query)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main test runner."""
    print("🚀 Vector Store Retrieval Testing")
    print("=" * 80)
    
    runner = VectorStoreTestRunner()
    
    # Show available test modes
    print("\nAvailable test modes:")
    print("1. Schema coverage analysis")
    print("2. SQL generation scenarios")
    print("3. Interactive mode")
    print("4. Predefined query comparison")
    
    mode = input("\nSelect mode (1-4) or press Enter for all: ").strip()
    
    if mode == "1" or not mode:
        runner.analyze_schema_coverage()
    
    if mode == "2" or not mode:
        runner.test_sql_generation_scenarios()
    
    if mode == "4" or not mode:
        # Test some common queries
        test_queries = [
            "users table",
            "orders information", 
            "user orders relationship",
            "order items details",
            "user email and status",
            "total orders per user"
        ]
        runner.test_multiple_queries(test_queries)
    
    if mode == "3":
        runner.run_interactive_mode()


if __name__ == "__main__":
    main()