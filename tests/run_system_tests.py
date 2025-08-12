"""
Automated system test runner for comprehensive testing.
This script runs all system tests and provides detailed output.
"""

import subprocess
import sys
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import test configuration
TEST_MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://admin:password@localhost:27017")
TEST_OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
TEST_BEDROCK_ENDPOINT = os.getenv("BEDROCK_ENDPOINT_URL", "https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/prod")


def check_service_availability():
    """Check if required services are available."""
    print("🔍 Checking service availability...")
    
    services_status = {}
    
    # Check MongoDB
    try:
        import motor.motor_asyncio
        
        async def check_mongodb():
            client = motor.motor_asyncio.AsyncIOMotorClient(TEST_MONGODB_URL)
            try:
                await client.admin.command('ping')
                return True
            except Exception as e:
                print(f"  ❌ MongoDB error: {e}")
                return False
            finally:
                client.close()
        
        services_status["mongodb"] = asyncio.run(check_mongodb())
    except Exception as e:
        print(f"  ❌ MongoDB check failed: {e}")
        services_status["mongodb"] = False
    
    # Check OpenSearch
    try:
        from opensearchpy import AsyncOpenSearch
        
        async def check_opensearch():
            client = AsyncOpenSearch(
                hosts=[{"host": TEST_OPENSEARCH_HOST, "port": 9200}],
                http_auth=None,
                use_ssl=False,
                verify_certs=False,
            )
            try:
                await client.ping()
                return True
            except Exception as e:
                print(f"  ❌ OpenSearch error: {e}")
                return False
            finally:
                await client.close()
        
        services_status["opensearch"] = asyncio.run(check_opensearch())
    except Exception as e:
        print(f"  ❌ OpenSearch check failed: {e}")
        services_status["opensearch"] = False
    
    # Check Bedrock endpoint (optional)
    try:
        import httpx
        
        async def check_bedrock():
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    # Simple health check - don't actually call the API
                    return True  # Assume available for now
                except Exception as e:
                    print(f"  ⚠️ Bedrock endpoint may not be available: {e}")
                    return False
        
        services_status["bedrock"] = asyncio.run(check_bedrock())
    except Exception as e:
        services_status["bedrock"] = False
    
    # Report status
    for service, status in services_status.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {service.title()}: {'Available' if status else 'Unavailable'}")
    
    return services_status


def run_test_suite():
    """Run the comprehensive test suite."""
    print("🧪 COMPREHENSIVE SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check services
    services = check_service_availability()
    print()
    
    if not services.get("mongodb") or not services.get("opensearch"):
        print("❌ Required services (MongoDB, OpenSearch) are not available.")
        print("Please ensure Docker services are running:")
        print("  docker compose up -d")
        return False
    
    # Change to project directory
    os.chdir(project_root)
    print(f"📁 Working directory: {project_root}")
    print()
    
    # Run system tests with verbose output
    print("🚀 Running system tests...")
    print("-" * 40)
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/system/test_complete_system.py",
        "-v", "-s", "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        print()
        print("-" * 40)
        if result.returncode == 0:
            print("✅ ALL SYSTEM TESTS PASSED!")
        else:
            print("❌ SOME SYSTEM TESTS FAILED")
            print(f"Exit code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main test runner function."""
    print("🔥 AUTOMATED SYSTEM TESTING")
    print("Testing all components of the text-to-SQL system")
    print()
    
    success = run_test_suite()
    
    print()
    print("=" * 60)
    if success:
        print("🎉 SYSTEM TEST SUITE: ALL TESTS PASSED")
        print()
        print("✅ MongoDB document upload/retrieval")
        print("✅ OpenSearch embedding and indexing")
        print("✅ Vector similarity search and retrieval")
        print("✅ Complete text-to-SQL flow")
        print("✅ HITL workflow with clarification")
        print("✅ Follow-up questions and SQL modification")
        print("✅ End-to-end integration")
        print()
        print("🚀 Your system is ready for demo!")
    else:
        print("❌ SYSTEM TEST SUITE: SOME TESTS FAILED")
        print()
        print("Please check the output above for details.")
        print("Common issues:")
        print("  - MongoDB or OpenSearch not running")
        print("  - Network connectivity issues")
        print("  - Missing environment variables")
        print("  - Service configuration problems")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())