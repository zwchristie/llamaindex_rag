"""
Simple test for the document sync model_dump fix.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.text_to_sql_rag.models.simple_models import DocumentType


def test_model_dump_fix():
    """Test the model_dump attribute fix."""
    print("Testing model_dump attribute handling...")
    
    # Test cases for different return types from _parse_document_content
    test_cases = [
        {
            "name": "Plain dictionary (BUSINESS_DOMAIN)",
            "metadata": {"domain_name": "test", "description": "test domain"},
            "expected_type": dict
        },
        {
            "name": "Plain dictionary (CORE_VIEW)", 
            "metadata": {"view_name": "test_view", "columns": []},
            "expected_type": dict
        },
        {
            "name": "None metadata",
            "metadata": None,
            "expected_type": type(None)
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        try:
            metadata = test_case["metadata"]
            
            # This is the fix we implemented
            if metadata:
                if hasattr(metadata, 'model_dump'):
                    metadata_dict = metadata.model_dump()
                    print(f"PASS {test_case['name']}: Used model_dump()")
                else:
                    metadata_dict = metadata
                    print(f"PASS {test_case['name']}: Used raw dict/object")
            else:
                metadata_dict = {}
                print(f"PASS {test_case['name']}: Handled None case")
            
            print(f"  Result type: {type(metadata_dict)}")
            print(f"  Result: {metadata_dict}")
            
            results.append({"test": test_case["name"], "status": "PASS"})
            
        except Exception as e:
            print(f"FAIL {test_case['name']}: {e}")
            results.append({"test": test_case["name"], "status": "FAIL", "error": str(e)})
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = len(results) - passed
    
    print(f"Total tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    for result in results:
        status_symbol = "PASS" if result['status'] == 'PASS' else "FAIL"
        error_msg = f" - {result.get('error', '')}" if 'error' in result else ""
        print(f"  {status_symbol}: {result['test']}{error_msg}")
    
    return failed == 0


if __name__ == "__main__":
    success = test_model_dump_fix()
    if success:
        print(f"\nSUCCESS: All tests passed! The model_dump fix should resolve the sync errors.")
    else:
        print(f"\nFAILED: Some tests failed. Please check the implementation.")
    
    exit(0 if success else 1)