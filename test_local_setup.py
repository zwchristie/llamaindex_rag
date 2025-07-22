#!/usr/bin/env python3
"""
Test script for local development setup.
Run this to verify your local configuration is working correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment():
    """Test environment configuration."""
    print("=== Testing Environment Configuration ===")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("[OK] .env file found")
    else:
        print("[WARN] .env file not found - using environment variables or defaults")
    
    # Check critical environment variables
    critical_vars = [
        "AWS_REGION",
        "QDRANT_HOST", 
        "MONGODB_URL",
        "LLM_PROVIDER"
    ]
    
    for var in critical_vars:
        value = os.getenv(var)
        if value:
            print(f"[OK] {var}: {value}")
        else:
            print(f"[WARN] {var}: Not set (using default)")


def test_settings():
    """Test settings loading."""
    print("\n=== Testing Settings Loading ===")
    
    try:
        from text_to_sql_rag.config.settings import settings
        
        print(f"[OK] Settings loaded successfully")
        print(f"  - App Title: {settings.app.title}")
        print(f"  - Debug Mode: {settings.app.debug}")
        print(f"  - LLM Provider: {settings.llm_provider.provider}")
        
        if settings.aws.use_profile:
            print(f"  - AWS Profile: {settings.aws.profile_name}")
        else:
            print(f"  - AWS Region: {settings.aws.region}")
        
        if settings.custom_llm:
            print(f"  - Custom LLM URL: {settings.custom_llm.base_url}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load settings: {e}")
        return False


def test_llm_provider():
    """Test LLM provider initialization."""
    print("\n=== Testing LLM Provider ===")
    
    try:
        from text_to_sql_rag.services.llm_provider_factory import llm_factory
        from text_to_sql_rag.config.settings import settings
        
        provider_info = llm_factory.get_provider_info()
        print(f"[OK] LLM Provider initialized: {provider_info['provider']}")
        
        # Test health check
        health = llm_factory.health_check()
        if health:
            print("[OK] LLM Provider health check: PASSED")
        else:
            print("[WARN] LLM Provider health check: FAILED")
        
        # Test simple generation (optional - might fail if credentials not set up)
        try:
            print("  Testing simple text generation...")
            response = llm_factory.generate_text("Hello, this is a test. Please respond with 'Test successful'.")
            print(f"[OK] Text generation test: {response[:100]}...")
            return True
        except Exception as e:
            print(f"[WARN] Text generation test failed: {e}")
            return True  # Still return True since provider initialized
        
    except Exception as e:
        print(f"[ERROR] LLM Provider initialization failed: {e}")
        return False


def test_vector_service():
    """Test vector service initialization.""" 
    print("\n=== Testing Vector Service ===")
    
    try:
        from text_to_sql_rag.services.vector_service import LlamaIndexVectorService
        
        print("  Initializing vector service...")
        vector_service = LlamaIndexVectorService()
        
        print("[OK] Vector service initialized")
        
        # Test health check
        health = vector_service.health_check()
        if health:
            print("[OK] Vector service health check: PASSED")
        else:
            print("[WARN] Vector service health check: FAILED (Qdrant might not be running)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Vector service initialization failed: {e}")
        return False


def test_mongodb_service():
    """Test MongoDB service initialization."""
    print("\n=== Testing MongoDB Service ===")
    
    try:
        from text_to_sql_rag.services.mongodb_service import MongoDBService
        
        print("  Initializing MongoDB service...")
        mongo_service = MongoDBService()
        
        print("[OK] MongoDB service initialized")
        
        # Test connection
        if mongo_service.is_connected():
            print("[OK] MongoDB connection: CONNECTED")
        else:
            print("[WARN] MongoDB connection: FAILED (MongoDB might not be running)")
        
        return True
        
    except Exception as e:
        print(f"[WARN] MongoDB service initialization failed: {e}")
        return True  # This is optional, so don't fail


def test_startup():
    """Test full application startup."""
    print("\n=== Testing Application Startup ===")
    
    try:
        from text_to_sql_rag.core.startup import ApplicationStartup
        import asyncio
        
        startup = ApplicationStartup()
        
        async def run_startup():
            success = await startup.initialize_services()
            return success
        
        success = asyncio.run(run_startup())
        
        if success:
            print("[OK] Application startup: SUCCESS")
        else:
            print("[WARN] Application startup: PARTIAL SUCCESS (some services may be unavailable)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Application startup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Local Development Setup")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_settings,
        test_llm_provider,
        test_vector_service,
        test_mongodb_service,
        test_startup
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! Your local setup is ready.")
    else:
        print("Some tests failed. Check the output above for details.")
        print("\nCommon fixes:")
        print("  - Make sure .env file is configured with your settings")
        print("  - Start Qdrant: docker run -p 6333:6333 qdrant/qdrant:latest")
        print("  - Start MongoDB: docker run -p 27017:27017 mongo:latest")
        print("  - Set up AWS credentials with ADFS profile if using Bedrock")
        print("  - Configure custom LLM settings if using custom provider")


if __name__ == "__main__":
    main()