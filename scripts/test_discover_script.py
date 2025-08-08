#!/usr/bin/env python3
"""
Quick test to verify discover_and_migrate_metadata.py creates embedding documents.

This script runs the discovery in dry-run mode to see what would be created.

Usage:
    python scripts/test_discover_script.py
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.discover_and_migrate_metadata import DynamicMetadataDiscoverer


def test_discovery():
    """Test metadata discovery process."""
    print("🔍 Testing metadata discovery process...")
    print("=" * 50)
    
    try:
        discoverer = DynamicMetadataDiscoverer()
        
        # Test view discovery
        print("📋 Discovering view metadata...")
        view_metadata = discoverer.discover_view_metadata()
        print(f"Found {len(view_metadata)} views")
        
        # Test report discovery
        print("\n📊 Discovering report metadata...")
        report_metadata = discoverer.discover_report_metadata()
        print(f"Found {len(report_metadata)} reports")
        
        # Test lookup discovery
        print("\n🔍 Discovering lookup metadata...")
        lookup_metadata = discoverer.discover_lookup_metadata()
        print(f"Found {len(lookup_metadata)} lookups")
        
        # Show combined results
        all_metadata = view_metadata + report_metadata + lookup_metadata
        print(f"\n📈 Total metadata items: {len(all_metadata)}")
        
        # Show sample content for first item
        if all_metadata:
            print(f"\n📄 Sample metadata structure:")
            sample = all_metadata[0]
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items - {value[:2] if value else '[]'}")
                elif isinstance(value, str) and len(value) > 50:
                    print(f"  {key}: {value[:50]}...")
                else:
                    print(f"  {key}: {value}")
        
        # Test embedding content generation
        if all_metadata:
            print(f"\n📝 Sample embedding content:")
            sample_content = discoverer._generate_embedding_content(all_metadata[0])
            print(sample_content[:300] + "..." if len(sample_content) > 300 else sample_content)
        
        # Test business domain discovery
        print(f"\n🏢 Business domains: {len(discoverer.business_domains)}")
        for domain in discoverer.business_domains[:3]:
            print(f"  - {domain['title']}: {domain['summary'][:60]}...")
        
        print(f"\n✅ Discovery test completed successfully!")
        print(f"Ready to run: python scripts/discover_and_migrate_metadata.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Discovery test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_discovery()
    if not success:
        sys.exit(1)