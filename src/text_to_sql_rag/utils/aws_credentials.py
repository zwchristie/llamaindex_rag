"""AWS credentials management for local development with assumed roles."""

import os
import configparser
from typing import Optional, Dict, Any
from pathlib import Path
import structlog
from datetime import datetime

logger = structlog.get_logger(__name__)


class AWSCredentialsManager:
    """Manages AWS credentials for local development with assumed roles."""
    
    def __init__(self, profile_name: str = "adfs"):
        self.profile_name = profile_name
        self.credentials_file = self._get_credentials_file_path()
        
    def _get_credentials_file_path(self) -> Path:
        """Get the AWS credentials file path."""
        # Try common locations for AWS credentials
        aws_credentials_paths = [
            Path.home() / ".aws" / "credentials",
            Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "")),
        ]
        
        for path in aws_credentials_paths:
            if path and path.exists():
                logger.info("Found AWS credentials file", path=str(path))
                return path
        
        # Default location
        default_path = Path.home() / ".aws" / "credentials"
        logger.info("Using default AWS credentials file path", path=str(default_path))
        return default_path
    
    def get_credentials(self) -> Optional[Dict[str, str]]:
        """Get credentials from the AWS credentials file for the specified profile."""
        try:
            if not self.credentials_file.exists():
                logger.error("AWS credentials file not found", path=str(self.credentials_file))
                return None
            
            config = configparser.ConfigParser()
            config.read(self.credentials_file)
            
            if self.profile_name not in config:
                logger.error("Profile not found in credentials file", 
                           profile=self.profile_name, 
                           available_profiles=list(config.sections()))
                return None
            
            profile_section = config[self.profile_name]
            
            # Extract credentials from the profile
            credentials = {}
            
            # Check for standard credential fields
            credential_fields = {
                'aws_access_key_id': 'aws_access_key_id',
                'aws_secret_access_key': 'aws_secret_access_key', 
                'aws_session_token': 'aws_session_token',
                'region': 'region',
                'aws_assumed_role_arn': 'aws_assumed_role_arn',
                'aws_credential_expiration': 'aws_credential_expiration'
            }
            
            for config_key, cred_key in credential_fields.items():
                if config_key in profile_section:
                    credentials[cred_key] = profile_section[config_key]
            
            # Validate required fields
            required_fields = ['aws_access_key_id', 'aws_secret_access_key']
            missing_fields = [field for field in required_fields if field not in credentials]
            
            if missing_fields:
                logger.error("Missing required credential fields", 
                           missing=missing_fields, 
                           profile=self.profile_name)
                return None
            
            # Check if credentials are expired
            if self._are_credentials_expired(credentials):
                logger.warning("AWS credentials appear to be expired", profile=self.profile_name)
                # Don't return None here - let AWS SDK handle the error
            
            logger.info("Successfully loaded AWS credentials", 
                       profile=self.profile_name,
                       has_session_token=bool(credentials.get('aws_session_token')),
                       expiration=credentials.get('aws_credential_expiration', 'not_set'))
            
            return credentials
            
        except Exception as e:
            logger.error("Failed to read AWS credentials", error=str(e), profile=self.profile_name)
            return None
    
    def _are_credentials_expired(self, credentials: Dict[str, str]) -> bool:
        """Check if the credentials are expired based on expiration timestamp."""
        expiration_str = credentials.get('aws_credential_expiration')
        if not expiration_str:
            return False
        
        try:
            # Parse the expiration timestamp
            # Common formats: ISO 8601, Unix timestamp, etc.
            if expiration_str.isdigit():
                # Unix timestamp
                expiration_time = datetime.fromtimestamp(int(expiration_str))
            else:
                # Try to parse as ISO format
                expiration_time = datetime.fromisoformat(expiration_str.replace('Z', '+00:00'))
            
            current_time = datetime.now()
            if expiration_time.tzinfo:
                # If expiration has timezone info, make current_time timezone aware
                from datetime import timezone
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            return current_time >= expiration_time
            
        except (ValueError, TypeError) as e:
            logger.warning("Could not parse credential expiration time", 
                         expiration=expiration_str, error=str(e))
            return False
    
    def get_session_config(self) -> Optional[Dict[str, Any]]:
        """Get boto3 session configuration for the credentials."""
        credentials = self.get_credentials()
        if not credentials:
            return None
        
        session_config = {
            'aws_access_key_id': credentials['aws_access_key_id'],
            'aws_secret_access_key': credentials['aws_secret_access_key'],
            'region_name': credentials.get('region', 'us-east-1')
        }
        
        # Add session token if available
        if 'aws_session_token' in credentials:
            session_config['aws_session_token'] = credentials['aws_session_token']
        
        return session_config
    
    def validate_credentials(self) -> bool:
        """Validate that credentials are present and not expired."""
        credentials = self.get_credentials()
        if not credentials:
            return False
        
        return not self._are_credentials_expired(credentials)
    
    def get_credential_info(self) -> Dict[str, Any]:
        """Get information about the current credentials (for debugging)."""
        credentials = self.get_credentials()
        if not credentials:
            return {
                "status": "not_found",
                "profile": self.profile_name,
                "credentials_file": str(self.credentials_file),
                "file_exists": self.credentials_file.exists()
            }
        
        info = {
            "status": "found",
            "profile": self.profile_name,
            "credentials_file": str(self.credentials_file),
            "file_exists": self.credentials_file.exists(),
            "has_access_key": bool(credentials.get('aws_access_key_id')),
            "has_secret_key": bool(credentials.get('aws_secret_access_key')),
            "has_session_token": bool(credentials.get('aws_session_token')),
            "region": credentials.get('region', 'not_set'),
            "assumed_role_arn": credentials.get('aws_assumed_role_arn', 'not_set'),
            "expiration": credentials.get('aws_credential_expiration', 'not_set'),
            "is_expired": self._are_credentials_expired(credentials)
        }
        
        return info