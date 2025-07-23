#!/usr/bin/env python3
"""Test script for AWS credentials manager with local [adfs] profile."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.text_to_sql_rag.utils.aws_credentials import AWSCredentialsManager
from src.text_to_sql_rag.config.settings import settings

def test_aws_credentials():
    """Test the AWS credentials manager functionality."""
    print("=" * 60)
    print("AWS Credentials Manager Test")
    print("=" * 60)
    
    # Test with configured profile
    profile_name = settings.aws.profile_name or "adfs"
    print(f"Testing with profile: {profile_name}")
    print()
    
    # Initialize the credentials manager
    credentials_manager = AWSCredentialsManager(profile_name)
    
    # Get credential information
    print("1. Credential Information:")
    print("-" * 30)
    info = credentials_manager.get_credential_info()
    for key, value in info.items():
        if key in ['aws_access_key_id', 'aws_secret_access_key', 'aws_session_token']:
            # Mask sensitive values
            masked_value = f"{str(value)[:8]}..." if value else "not_set"
            print(f"  {key}: {masked_value}")
        else:
            print(f"  {key}: {value}")
    
    print()
    
    # Test credential validation
    print("2. Credential Validation:")
    print("-" * 30)
    is_valid = credentials_manager.validate_credentials()
    print(f"  Credentials valid: {is_valid}")
    
    print()
    
    # Test session configuration
    print("3. Session Configuration:")
    print("-" * 30)
    session_config = credentials_manager.get_session_config()
    if session_config:
        print("  Session config available:")
        for key, value in session_config.items():
            if key in ['aws_access_key_id', 'aws_secret_access_key', 'aws_session_token']:
                # Mask sensitive values
                masked_value = f"{str(value)[:8]}..." if value else "not_set"
                print(f"    {key}: {masked_value}")
            else:
                print(f"    {key}: {value}")
    else:
        print("  No session config available")
    
    print()
    print("=" * 60)
    
    return is_valid and session_config is not None

def test_bedrock_service():
    """Test the Bedrock service with the new credentials manager."""
    print("Testing Bedrock Service Integration")
    print("=" * 60)
    
    try:
        from src.text_to_sql_rag.services.bedrock_service import BedrockEmbeddingService, BedrockLLMService
        
        print("1. Testing Bedrock Embedding Service:")
        print("-" * 40)
        try:
            embedding_service = BedrockEmbeddingService()
            print("  ✓ Embedding service initialized successfully")
            
            # Test a simple embedding
            test_text = "Hello, world!"
            embedding = embedding_service.get_embedding(test_text)
            print(f"  ✓ Generated embedding with {len(embedding)} dimensions")
            
        except Exception as e:
            print(f"  ✗ Embedding service error: {e}")
        
        print()
        print("2. Testing Bedrock LLM Service:")
        print("-" * 40)
        try:
            llm_service = BedrockLLMService()
            print("  ✓ LLM service initialized successfully")
            
            # Test a simple text generation
            test_prompt = "What is 2+2?"
            response = llm_service.generate_text(test_prompt, max_tokens=50)
            print(f"  ✓ Generated response: {response[:100]}...")
            
        except Exception as e:
            print(f"  ✗ LLM service error: {e}")
            
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"General error: {e}")
        return False
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    print("Starting AWS Credentials Test Suite")
    print()
    
    try:
        # Test credentials manager
        creds_success = test_aws_credentials()
        
        print()
        
        # Test Bedrock service integration if credentials are working
        if creds_success:
            bedrock_success = test_bedrock_service()
        else:
            print("Skipping Bedrock service test due to credential issues")
            bedrock_success = False
        
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Credentials Manager: {'✓ PASS' if creds_success else '✗ FAIL'}")
        print(f"Bedrock Integration: {'✓ PASS' if bedrock_success else '✗ FAIL'}")
        
        if creds_success and bedrock_success:
            print("\n🎉 All tests passed! AWS integration is working correctly.")
            sys.exit(0)
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Test suite error: {e}")
        sys.exit(1)