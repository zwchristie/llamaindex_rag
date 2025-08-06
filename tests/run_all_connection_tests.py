#!/usr/bin/env python3
"""
Master Connection Test Runner

Runs all connection tests for the text-to-SQL RAG application.
Tests MongoDB, OpenSearch, Bedrock LLM, and Bedrock Embeddings.
Provides comprehensive connectivity validation and configuration guidance.
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Add current directory to path for importing test modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from text_to_sql_rag.config.settings import settings
except ImportError as e:
    print(f"❌ ERROR: Cannot import application settings: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class MasterTestRunner:
    """Run all connection tests and provide comprehensive results."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Define test modules
        self.test_modules = {
            "mongodb": {
                "name": "MongoDB Connection Test",
                "script": "test_mongodb_connection.py",
                "description": "Tests MongoDB connectivity, authentication, and CRUD operations"
            },
            "opensearch": {
                "name": "OpenSearch Connection Test",
                "script": "test_opensearch_connection.py", 
                "description": "Tests OpenSearch connectivity, indexing, and vector search"
            },
            "bedrock_llm": {
                "name": "Bedrock LLM Connection Test",
                "script": "test_bedrock_llm.py",
                "description": "Tests AWS Bedrock LLM connectivity and text generation"
            },
            "bedrock_embedding": {
                "name": "Bedrock Embedding Connection Test",
                "script": "test_bedrock_embedding.py",
                "description": "Tests AWS Bedrock embedding connectivity and vector generation"
            }
        }
    
    def print_header(self):
        """Print test suite header."""
        print("🧪 Text-to-SQL RAG Application - Connection Test Suite")
        print("=" * 80)
        print("This comprehensive test suite validates connectivity to all external services")
        print("required for the text-to-SQL RAG application to function properly.")
        print()
        
        # Print current configuration summary
        print("📋 Current Configuration:")
        print(f"   MongoDB:     {settings.mongodb.url}")
        print(f"   OpenSearch:  {settings.opensearch.host}:{settings.opensearch.port} (SSL: {settings.opensearch.use_ssl})")
        print(f"   Bedrock Endpoint: {settings.bedrock_endpoint_url or 'Not configured'}")
        print(f"   Bedrock SSL Verify: {settings.bedrock_endpoint_verify_ssl}")
        print(f"   LLM Model:   {settings.aws.llm_model}")
        print(f"   Embed Model: {settings.aws.embedding_model}")
        print(f"   LLM Provider: {settings.llm_provider.provider}")
        print()
    
    def run_single_test(self, test_key: str, test_info: Dict[str, str]) -> Dict[str, Any]:
        """Run a single test module."""
        script_path = os.path.join(current_dir, test_info["script"])
        
        if not os.path.exists(script_path):
            return {
                "success": False,
                "error": f"Test script not found: {test_info['script']}",
                "duration": 0,
                "stdout": "",
                "stderr": ""
            }
        
        print(f"🔍 Running {test_info['name']}...")
        print(f"   {test_info['description']}")
        
        try:
            start_time = time.time()
            
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.path.dirname(current_dir)  # Run from project root
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "return_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test timed out after 5 minutes",
                "duration": 300,
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to run test: {str(e)}",
                "duration": 0,
                "stdout": "",
                "stderr": ""
            }
    
    def run_all_tests(self, selected_tests: Optional[List[str]] = None):
        """Run all connection tests."""
        self.start_time = time.time()
        
        # Determine which tests to run
        if selected_tests:
            tests_to_run = {k: v for k, v in self.test_modules.items() if k in selected_tests}
        else:
            tests_to_run = self.test_modules
        
        print(f"🚀 Starting connection tests ({len(tests_to_run)} tests)...")
        print("-" * 80)
        print()
        
        for test_key, test_info in tests_to_run.items():
            result = self.run_single_test(test_key, test_info)
            self.results[test_key] = result
            
            # Print immediate result
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration_str = f"{result['duration']:.1f}s"
            print(f"{status} {test_info['name']} ({duration_str})")
            
            if not result["success"]:
                if result.get("error"):
                    print(f"     Error: {result['error']}")
                if result.get("stderr") and result["stderr"].strip():
                    # Show first few lines of stderr
                    stderr_lines = result["stderr"].strip().split('\n')
                    for line in stderr_lines[:3]:
                        if line.strip():
                            print(f"     {line}")
                    if len(stderr_lines) > 3:
                        print(f"     ... ({len(stderr_lines) - 3} more lines)")
            
            print()
        
        self.end_time = time.time()
    
    def print_detailed_results(self):
        """Print detailed test results."""
        total_duration = self.end_time - self.start_time
        passed_tests = sum(1 for result in self.results.values() if result["success"])
        total_tests = len(self.results)
        
        print("=" * 80)
        print("📊 DETAILED TEST RESULTS")
        print("=" * 80)
        
        for test_key, result in self.results.items():
            test_info = self.test_modules[test_key]
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            
            print(f"\n{test_info['name']}: {status}")
            print(f"   Duration: {result['duration']:.1f}s")
            
            if result["success"]:
                # Try to extract test statistics from stdout
                stdout_lines = result["stdout"].split('\n')
                for line in stdout_lines:
                    if "Test Results:" in line or "tests passed" in line:
                        print(f"   {line.strip()}")
            else:
                if result.get("error"):
                    print(f"   Error: {result['error']}")
                print(f"   Return Code: {result.get('return_code', 'N/A')}")
        
        # Overall summary
        print("\n" + "=" * 80)
        print("🎯 OVERALL SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed:      {passed_tests} ✅")
        print(f"Failed:      {total_tests - passed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Duration: {total_duration:.1f}s")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Your application should be ready to use.")
        elif passed_tests > 0:
            print(f"\n⚠️  PARTIAL SUCCESS: {total_tests - passed_tests} service(s) need attention.")
            print("   The application may work with limited functionality.")
        else:
            print("\n❌ ALL TESTS FAILED: Please check your configuration and connectivity.")
    
    def print_configuration_guidance(self):
        """Print configuration guidance for failed tests."""
        failed_tests = [k for k, v in self.results.items() if not v["success"]]
        
        if not failed_tests:
            return
        
        print("\n" + "=" * 80)
        print("🔧 CONFIGURATION GUIDANCE")
        print("=" * 80)
        
        guidance = {
            "mongodb": {
                "title": "MongoDB Configuration",
                "env_vars": [
                    "MONGODB_URL=mongodb://localhost:27017",
                    "MONGODB_DATABASE=text_to_sql_rag"
                ],
                "notes": [
                    "Ensure MongoDB is running and accessible",
                    "Check network connectivity to MongoDB host",
                    "Verify authentication credentials if required"
                ]
            },
            "opensearch": {
                "title": "OpenSearch Configuration", 
                "env_vars": [
                    "OPENSEARCH_HOST=localhost",
                    "OPENSEARCH_PORT=9200",
                    "OPENSEARCH_USE_SSL=true",
                    "OPENSEARCH_USERNAME=admin",
                    "OPENSEARCH_PASSWORD=admin"
                ],
                "notes": [
                    "Ensure OpenSearch/Elasticsearch is running and accessible",
                    "Check SSL/TLS configuration matches your setup",
                    "Verify authentication credentials",
                    "Ensure vector dimension (OPENSEARCH_VECTOR_SIZE) matches embedding model"
                ]
            },
            "bedrock_llm": {
                "title": "Bedrock Endpoint LLM Configuration",
                "env_vars": [
                    "BEDROCK_ENDPOINT_URL=https://your-endpoint.com/invokeBedrock/",
                    "BEDROCK_ENDPOINT_VERIFY_SSL=true",
                    "AWS_LLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "LLM_PROVIDER=bedrock"
                ],
                "notes": [
                    "Ensure BEDROCK_ENDPOINT_URL is configured for your endpoint",
                    "Set BEDROCK_ENDPOINT_VERIFY_SSL=false to disable SSL verification if needed",
                    "Verify endpoint has proper authentication and permissions",
                    "Check if the LLM model is supported by your endpoint",
                    "Endpoint must support the expected request/response format"
                ]
            },
            "bedrock_embedding": {
                "title": "Bedrock Endpoint Embedding Configuration",
                "env_vars": [
                    "BEDROCK_ENDPOINT_URL=https://your-endpoint.com/invokeBedrock/",
                    "BEDROCK_ENDPOINT_VERIFY_SSL=true",
                    "AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0",
                    "OPENSEARCH_VECTOR_SIZE=1024"
                ],
                "notes": [
                    "Ensure BEDROCK_ENDPOINT_URL is configured for your endpoint",
                    "Set BEDROCK_ENDPOINT_VERIFY_SSL=false to disable SSL verification if needed",
                    "Verify endpoint has proper authentication and permissions",
                    "Check if the embedding model is supported by your endpoint",
                    "Ensure vector size matches the embedding model dimensions"
                ]
            }
        }
        
        for test_key in failed_tests:
            if test_key in guidance:
                guide = guidance[test_key]
                print(f"\n🔧 {guide['title']}:")
                print("   Environment Variables:")
                for env_var in guide["env_vars"]:
                    print(f"     {env_var}")
                print("   Notes:")
                for note in guide["notes"]:
                    print(f"     • {note}")
    
    def save_results_to_file(self, filename: Optional[str] = None):
        """Save test results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"connection_test_results_{timestamp}.json"
        
        filepath = os.path.join(current_dir, filename)
        
        # Prepare results for JSON serialization
        json_results = {
            "test_run_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_duration": self.end_time - self.start_time if self.end_time and self.start_time else 0,
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results.values() if r["success"])
            },
            "configuration": {
                "mongodb_url": settings.mongodb.url,
                "opensearch_host": settings.opensearch.host,
                "opensearch_port": settings.opensearch.port,
                "bedrock_endpoint_url": settings.bedrock_endpoint_url,
                "bedrock_endpoint_verify_ssl": settings.bedrock_endpoint_verify_ssl,
                "llm_model": settings.aws.llm_model,
                "embedding_model": settings.aws.embedding_model,
                "llm_provider": settings.llm_provider.provider
            },
            "test_results": self.results
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Test results saved to: {filepath}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {str(e)}")


def main():
    """Main function to run all connection tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run connection tests for text-to-SQL RAG application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_connection_tests.py                    # Run all tests
  python run_all_connection_tests.py --tests mongodb    # Run only MongoDB test
  python run_all_connection_tests.py --save-results     # Save results to JSON file
  python run_all_connection_tests.py --quick           # Skip detailed output
        """
    )
    
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["mongodb", "opensearch", "bedrock_llm", "bedrock_embedding"],
        help="Run only specific tests (default: run all tests)"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to JSON file"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - minimal output"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("src/text_to_sql_rag"):
        print("❌ ERROR: Please run this script from the project root directory")
        sys.exit(1)
    
    # Create and run test suite
    runner = MasterTestRunner()
    
    if not args.quick:
        runner.print_header()
    
    # Run tests
    runner.run_all_tests(selected_tests=args.tests)
    
    # Print results
    if not args.quick:
        runner.print_detailed_results()
        runner.print_configuration_guidance()
    else:
        # Quick summary
        passed = sum(1 for r in runner.results.values() if r["success"])
        total = len(runner.results)
        print(f"\n📊 Quick Summary: {passed}/{total} tests passed")
    
    # Save results if requested
    if args.save_results:
        runner.save_results_to_file()
    
    # Exit with appropriate code
    all_passed = all(result["success"] for result in runner.results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())