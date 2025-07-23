#!/usr/bin/env python3
"""Simple test script for AWS credentials manager with local [adfs] profile."""

import sys
import os
import configparser
from pathlib import Path
from datetime import datetime

def test_aws_credentials_simple():
    """Test AWS credentials manager functionality without dependencies."""
    print("=" * 60)
    print("Simple AWS Credentials Test")
    print("=" * 60)
    
    # Define profile name (the one user specified)
    profile_name = "adfs"
    
    print(f"Testing with profile: {profile_name}")
    print()
    
    # Find AWS credentials file
    credentials_paths = [
        Path.home() / ".aws" / "credentials",
        Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "")),
    ]
    
    credentials_file = None
    for path in credentials_paths:
        if path and path.exists():
            credentials_file = path
            print(f"Found AWS credentials file: {path}")
            break
    
    if not credentials_file:
        print("[ERROR] No AWS credentials file found")
        print("Expected locations:")
        for path in credentials_paths[:1]:  # Only show the standard location
            print(f"  - {path}")
        return False
    
    print()
    
    # Parse credentials file
    try:
        config = configparser.ConfigParser()
        config.read(credentials_file)
        
        if profile_name not in config:
            print(f"[ERROR] Profile '{profile_name}' not found in credentials file")
            print(f"Available profiles: {list(config.sections())}")
            return False
        
        profile_section = config[profile_name]
        
        # Check for required fields
        required_fields = ['aws_access_key_id', 'aws_secret_access_key']
        credentials = {}
        
        for field in required_fields:
            if field in profile_section:
                credentials[field] = profile_section[field]
        
        # Optional fields
        optional_fields = ['aws_session_token', 'region', 'aws_assumed_role_arn', 'aws_credential_expiration']
        for field in optional_fields:
            if field in profile_section:
                credentials[field] = profile_section[field]
        
        print("1. Credential Fields Found:")
        print("-" * 30)
        for field in required_fields + optional_fields:
            if field in credentials:
                if field in ['aws_access_key_id', 'aws_secret_access_key', 'aws_session_token']:
                    # Mask sensitive values
                    value = credentials[field]
                    masked_value = f"{value[:8]}..." if len(value) > 8 else "****"
                    print(f"  [OK] {field}: {masked_value}")
                else:
                    print(f"  [OK] {field}: {credentials[field]}")
            else:
                print(f"  - {field}: not_found")
        
        print()
        
        # Check if all required fields are present
        missing_required = [field for field in required_fields if field not in credentials]
        if missing_required:
            print(f"[ERROR] Missing required fields: {missing_required}")
            return False
        
        print("2. Credential Validation:")
        print("-" * 30)
        print("  [OK] All required fields present")
        
        # Check expiration if present
        expiration_str = credentials.get('aws_credential_expiration')
        if expiration_str:
            try:
                if expiration_str.isdigit():
                    # Unix timestamp
                    expiration_time = datetime.fromtimestamp(int(expiration_str))
                else:
                    # Try to parse as ISO format
                    expiration_time = datetime.fromisoformat(expiration_str.replace('Z', '+00:00'))
                
                current_time = datetime.now()
                if expiration_time.tzinfo:
                    from datetime import timezone
                    current_time = current_time.replace(tzinfo=timezone.utc)
                
                is_expired = current_time >= expiration_time
                status = "[ERROR] EXPIRED" if is_expired else "[OK] Valid"
                print(f"  {status} Expires: {expiration_time}")
                
                if is_expired:
                    print("  [WARNING] Credentials appear to be expired!")
                    
            except Exception as e:
                print(f"  [WARNING] Could not parse expiration: {e}")
        else:
            print("  - No expiration time found")
        
        print()
        
        # Build session config
        print("3. Session Configuration:")
        print("-" * 30)
        session_config = {
            'aws_access_key_id': credentials['aws_access_key_id'],
            'aws_secret_access_key': credentials['aws_secret_access_key'],
            'region_name': credentials.get('region', 'us-east-1')
        }
        
        if 'aws_session_token' in credentials:
            session_config['aws_session_token'] = credentials['aws_session_token']
        
        print("  Session config ready:")
        for key, value in session_config.items():
            if key in ['aws_access_key_id', 'aws_secret_access_key', 'aws_session_token']:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "****"
                print(f"    {key}: {masked_value}")
            else:
                print(f"    {key}: {value}")
        
        print()
        print("[SUCCESS] AWS credentials test completed successfully!")
        print()
        print("Summary:")
        print(f"  - Profile: {profile_name}")
        print(f"  - Credentials file: {credentials_file}")
        print(f"  - Has session token: {'yes' if 'aws_session_token' in credentials else 'no'}")
        print(f"  - Region: {credentials.get('region', 'us-east-1 (default)')}")
        if credentials.get('aws_assumed_role_arn'):
            print(f"  - Assumed role: {credentials['aws_assumed_role_arn']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error reading credentials file: {e}")
        return False

if __name__ == "__main__":
    print("AWS Credentials Simple Test")
    print()
    
    success = test_aws_credentials_simple()
    
    print()
    print("=" * 60)
    
    if success:
        print("[SUCCESS] Test PASSED! Your AWS credentials are properly configured.")
        print()
        print("Next steps:")
        print("1. Ensure your AWS credentials are not expired")
        print("2. Verify Bedrock permissions on the assumed role")
        print("3. Test the full application")
        sys.exit(0)
    else:
        print("[ERROR] Test FAILED! Please check your AWS credentials configuration.")
        print()
        print("To fix:")
        print("1. Ensure you have run 'aws sts assume-role' recently")
        print("2. Check that your credentials file has the [adfs] profile")
        print("3. Verify all required fields are present")
        sys.exit(1)