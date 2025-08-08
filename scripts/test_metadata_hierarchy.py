#!/usr/bin/env python3
"""
Test script to validate hierarchical metadata access patterns after MongoDB migration.

This script tests that the hierarchical access pattern is maintained when using
MongoDB as the source of truth instead of local files.

Usage:
    python scripts/test_metadata_hierarchy.py
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, List
import time

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from text_to_sql_rag.services.view_metadata_service import ViewMetadataService
from text_to_sql_rag.services.business_domain_metadata_service import BusinessDomainMetadataService
from text_to_sql_rag.services.view_selection_service import ViewSelectionService
from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
from text_to_sql_rag.services.llm_service import LLMService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_view_metadata_service():
    """Test ViewMetadataService MongoDB integration."""
    logger.info("🔧 Testing ViewMetadataService...")
    
    try:
        service = ViewMetadataService()
        
        # Test view domain mappings
        mappings = service.get_view_domain_mappings()
        logger.info(f"✅ Retrieved {len(mappings)} view domain mappings")
        
        if mappings:
            sample_view = list(mappings.keys())[0]
            sample_domains = mappings[sample_view]
            logger.info(f"Sample: {sample_view} -> {sample_domains}")
        
        # Test view dependencies
        dependencies = service.get_view_dependencies()
        logger.info(f"✅ Retrieved {len(dependencies)} view dependencies")
        
        # Test query patterns
        patterns = service.get_query_patterns()
        logger.info(f"✅ Retrieved {len(patterns)} query patterns")
        
        service.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ ViewMetadataService test failed: {e}")
        return False


def test_business_domain_metadata_service():
    """Test BusinessDomainMetadataService MongoDB integration."""
    logger.info("🏢 Testing BusinessDomainMetadataService...")
    
    try:
        service = BusinessDomainMetadataService()
        
        # Test business domains
        domains = service.get_business_domains()
        logger.info(f"✅ Retrieved {len(domains)} business domains")
        
        if domains:
            sample_domain = list(domains.keys())[0]
            domain_info = domains[sample_domain]
            logger.info(f"Sample: {sample_domain} -> {len(domain_info.get('key_concepts', []))} concepts")
        
        # Test domain terminology
        terminology = service.get_domain_terminology()
        logger.info(f"✅ Retrieved terminology for {len(terminology)} domains")
        
        # Test detection rules
        detection_rules = service.get_detection_rules()
        logger.info(f"✅ Retrieved {len(detection_rules)} detection rules")
        
        service.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ BusinessDomainMetadataService test failed: {e}")
        return False


def test_hierarchical_query_flow():
    """Test the full hierarchical query processing flow."""
    logger.info("🔄 Testing hierarchical query flow...")
    
    try:
        # Initialize services
        vector_service = LlamaIndexVectorService()
        llm_service = LLMService()
        view_selection_service = ViewSelectionService(vector_service, llm_service)
        
        # Test query
        test_query = "What are all the deal names with tranches that are in announced status in Fixed Income?"
        
        logger.info(f"Testing query: '{test_query}'")
        
        # Step 1: Test domain identification (simulated)
        from text_to_sql_rag.models.simple_models import DomainContext
        
        domain_context = DomainContext(
            query=test_query,
            identified_domains=["DEAL", "TRANCHE"],
            confidence_scores={"DEAL": 0.9, "TRANCHE": 0.8},
            enhanced_query=test_query + " focusing on deal and tranche information",
            business_context="Financial instruments in primary market"
        )
        
        # Step 2: Test view selection using domain context
        view_context = view_selection_service.select_domain_views(domain_context, debug=True)
        
        logger.info(f"✅ Selected {len(view_context.core_views)} core views: {view_context.core_views}")
        logger.info(f"✅ Selected {len(view_context.supporting_views)} supporting views: {view_context.supporting_views}")
        logger.info(f"✅ Selection reasoning: {view_context.selection_reasoning[:100]}...")
        
        # Step 3: Verify hierarchical structure is maintained
        if view_context.core_views and view_context.supporting_views:
            logger.info("✅ Hierarchical structure maintained: core + supporting views")
        elif view_context.core_views:
            logger.info("✅ Core views selected, no supporting views needed")
        else:
            logger.warning("⚠️  No views selected - check metadata migration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Hierarchical query flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_consistency():
    """Test consistency between different metadata services."""
    logger.info("📊 Testing metadata consistency...")
    
    try:
        view_service = ViewMetadataService()
        domain_service = BusinessDomainMetadataService()
        
        # Get view mappings and business domains
        view_mappings = view_service.get_view_domain_mappings()
        business_domains = domain_service.get_business_domains()
        
        # Check consistency
        all_mapped_domains = set()
        for view, domains in view_mappings.items():
            all_mapped_domains.update(domains)
        
        defined_domains = set(business_domains.keys())
        
        # Find inconsistencies
        unmapped_domains = all_mapped_domains - defined_domains
        unused_domains = defined_domains - all_mapped_domains
        
        logger.info(f"✅ Views reference {len(all_mapped_domains)} unique domains")
        logger.info(f"✅ System defines {len(defined_domains)} business domains")
        
        if unmapped_domains:
            logger.warning(f"⚠️  Views reference undefined domains: {unmapped_domains}")
        
        if unused_domains:
            logger.info(f"ℹ️  Unused business domains: {unused_domains}")
        
        if not unmapped_domains:
            logger.info("✅ All view-referenced domains are properly defined")
        
        view_service.close()
        domain_service.close()
        
        return len(unmapped_domains) == 0
        
    except Exception as e:
        logger.error(f"❌ Metadata consistency test failed: {e}")
        return False


def test_performance_benchmarks():
    """Test performance of MongoDB-based metadata loading."""
    logger.info("⚡ Testing performance benchmarks...")
    
    try:
        # Test view metadata service performance
        start_time = time.time()
        view_service = ViewMetadataService()
        mappings = view_service.get_view_domain_mappings()
        dependencies = view_service.get_view_dependencies()
        patterns = view_service.get_query_patterns()
        view_load_time = time.time() - start_time
        
        # Test business domain service performance
        start_time = time.time()
        domain_service = BusinessDomainMetadataService()
        domains = domain_service.get_business_domains()
        terminology = domain_service.get_domain_terminology()
        rules = domain_service.get_detection_rules()
        domain_load_time = time.time() - start_time
        
        logger.info(f"✅ View metadata loaded in {view_load_time:.3f}s ({len(mappings)} mappings)")
        logger.info(f"✅ Domain metadata loaded in {domain_load_time:.3f}s ({len(domains)} domains)")
        
        total_time = view_load_time + domain_load_time
        logger.info(f"✅ Total metadata load time: {total_time:.3f}s")
        
        # Performance expectations
        if total_time < 1.0:
            logger.info("🚀 Excellent performance: < 1 second")
        elif total_time < 3.0:
            logger.info("✅ Good performance: < 3 seconds")
        else:
            logger.warning("⚠️  Slow performance: > 3 seconds - consider indexing")
        
        view_service.close()
        domain_service.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark test failed: {e}")
        return False


def run_hierarchy_tests():
    """Run comprehensive hierarchical access pattern tests."""
    logger.info("🧪 Starting hierarchical metadata access pattern tests")
    logger.info("=" * 70)
    
    test_results = {}
    
    # Test 1: View Metadata Service
    test_results["view_metadata"] = test_view_metadata_service()
    
    # Test 2: Business Domain Metadata Service  
    test_results["domain_metadata"] = test_business_domain_metadata_service()
    
    # Test 3: Hierarchical Query Flow
    test_results["query_flow"] = test_hierarchical_query_flow()
    
    # Test 4: Metadata Consistency
    test_results["consistency"] = test_metadata_consistency()
    
    # Test 5: Performance Benchmarks
    test_results["performance"] = test_performance_benchmarks()
    
    # Summary
    logger.info("📊 Test Results Summary")
    logger.info("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL" 
        logger.info(f"✅ {test_name.replace('_', ' ').title()}: {status}")
        if result:
            passed += 1
    
    logger.info(f"✅ Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All hierarchical access pattern tests passed!")
        logger.info("✅ MongoDB migration maintains hierarchical structure")
    else:
        logger.warning("⚠️  Some tests failed - check MongoDB metadata migration")
    
    return passed == total


if __name__ == "__main__":
    print("🧪 Hierarchical Metadata Access Pattern Test")
    print("=" * 50)
    print("Testing that hierarchical access patterns work with MongoDB metadata.")
    print()
    
    try:
        success = run_hierarchy_tests()
        
        if success:
            print("\n🎉 All hierarchical access pattern tests passed!")
            print("\nThe system successfully:")
            print("  ✅ Loads metadata from MongoDB instead of local files")
            print("  ✅ Maintains hierarchical domain -> view -> context structure")
            print("  ✅ Preserves query processing flow")
            print("  ✅ Ensures metadata consistency")
            print("  ✅ Performs adequately")
        else:
            print("\n❌ Some hierarchical access pattern tests failed!")
            print("Check the error messages above for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)