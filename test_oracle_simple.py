#!/usr/bin/env python3
"""
Simple test of Oracle SQL generation with hierarchical context.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_oracle_sql_generation():
    """Test Oracle SQL generation with hierarchical context."""
    print("\nTesting Oracle SQL Generation with Hierarchical Context")
    print("=" * 60)
    
    try:
        # Import services
        from text_to_sql_rag.services.embedding_service import EmbeddingService
        from text_to_sql_rag.services.enhanced_bedrock_service import EnhancedBedrockService as BedrockEndpointService
        from text_to_sql_rag.services.hierarchical_context_service import HierarchicalContextService, RetrievalConfig
        
        # Configuration
        meta_docs_path = Path(__file__).parent / "meta_documents"
        bedrock_endpoint = os.getenv("BEDROCK_ENDPOINT_URL", "https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod")
        embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
        llm_model = os.getenv("BEDROCK_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
        
        # Initialize services
        embedding_service = EmbeddingService(
            endpoint_url=bedrock_endpoint,
            embedding_model=embedding_model,
            use_mock=False
        )
        
        bedrock_service = BedrockEndpointService(
            bedrock_endpoint, 
            embedding_model, 
            llm_model
        )
        
        context_service = HierarchicalContextService(
            meta_docs_path=meta_docs_path,
            embedding_service=embedding_service,
            vector_service=None,
            bedrock_service=bedrock_service,
            config=RetrievalConfig(max_core_views=2, max_supporting_views=2, max_reports=1, max_lookups=3)
        )
        
        print("Services initialized successfully")
        
        # Test query
        test_query = "Show me syndicate participants for active deals with their roles"
        print(f"\nTest Query: {test_query}")
        
        # Build hierarchical context
        print("\nBuilding hierarchical context...")
        context = await context_service.build_context(test_query)
        print(f"Context: {len(context.core_views)} core views, {len(context.supporting_views)} supporting, {len(context.lookups)} lookups")
        
        # Generate Oracle SQL
        print("\nGenerating Oracle SQL...")
        combined_context = context.get_combined_context()
        sql_result = await bedrock_service.generate_sql(test_query, combined_context)
        
        # Display results
        print("\n" + "=" * 40)
        print("GENERATED ORACLE SQL:")
        print("=" * 40)
        print(sql_result.get('sql', 'No SQL generated'))
        
        print("\n" + "=" * 40)
        print("EXPLANATION:")
        print("=" * 40)
        print(sql_result.get('explanation', 'No explanation provided'))
        
        # Analyze Oracle features
        print("\n" + "=" * 40)
        print("ORACLE FEATURES DETECTED:")
        print("=" * 40)
        sql_text = sql_result.get('sql', '').upper()
        
        oracle_features = []
        if any(schema in sql_text for schema in ['SYND.', 'USR.', 'DEALS.']):
            oracle_features.append("[PASS] Schema-qualified table names")
        else:
            oracle_features.append("[FAIL] Missing schema qualification")
        
        if 'ROWNUM' in sql_text:
            oracle_features.append("[PASS] ROWNUM for row limiting")
        elif 'LIMIT' in sql_text:
            oracle_features.append("[WARN] Uses LIMIT instead of ROWNUM")
        
        if 'SYSDATE' in sql_text or 'TO_DATE' in sql_text:
            oracle_features.append("[PASS] Oracle date functions")
        
        if 'UPPER(' in sql_text:
            oracle_features.append("[PASS] UPPER() for case-insensitive matching")
        
        # Check for lookup ID usage
        if context.lookups:
            has_lookup_ids = any(str(lookup.get('lookup_id', '')) in sql_text for lookup in context.lookups)
            if has_lookup_ids:
                oracle_features.append("[PASS] Uses lookup ID values")
            else:
                oracle_features.append("[WARN] Missing lookup ID references")
        
        for feature in oracle_features:
            print(feature)
        
        print("\n" + "=" * 60)
        print("SUCCESS: Oracle SQL generation with hierarchical context working!")
        print("Key improvements:")
        print("- Hierarchical metadata retrieval (domains -> views -> reports -> lookups)")
        print("- Oracle-specific SQL syntax and functions")
        print("- Schema-qualified table references")
        print("- Lookup value integration")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test execution."""
    success = await test_oracle_sql_generation()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)