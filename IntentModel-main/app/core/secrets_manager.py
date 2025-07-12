"""
AWS Secrets Manager integration for secure credential management.
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Any
from functools import lru_cache
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class SecretsManagerConfig(BaseSettings):
    """Configuration for AWS Secrets Manager."""
    
    aws_region: str = "us-east-1"
    secrets_prefix: str = "leadpoet"
    cache_ttl_seconds: int = 300  # 5 minutes cache
    environment: str = "production"
    application_name: str = "leadpoet-intent-model"
    
    class Config:
        env_prefix = "SECRETS_"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with environment variables if not explicitly set
        if not kwargs.get('environment'):
            self.environment = os.getenv('ENVIRONMENT', 'production')
        if not kwargs.get('application_name'):
            self.application_name = os.getenv('APPLICATION_NAME', 'leadpoet-intent-model')


class SecretsManager:
    """
    AWS Secrets Manager client for secure credential management.
    
    This class provides a secure way to retrieve and cache secrets
    from AWS Secrets Manager, following security best practices.
    """
    
    def __init__(self, config: Optional[SecretsManagerConfig] = None):
        """Initialize the Secrets Manager client."""
        self.config = config or SecretsManagerConfig()
        self._client = None
        self._cache = {}
        self._cache_timestamps = {}
        
    @property
    def client(self):
        """Lazy initialization of the AWS Secrets Manager client."""
        if self._client is None:
            try:
                self._client = boto3.client(
                    'secretsmanager',
                    region_name=self.config.aws_region
                )
                logger.info("AWS Secrets Manager client initialized successfully")
            except NoCredentialsError:
                logger.error("AWS credentials not found. Please configure AWS credentials.")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize AWS Secrets Manager client: {e}")
                raise
        return self._client
    
    def _is_cache_valid(self, secret_name: str) -> bool:
        """Check if the cached secret is still valid."""
        if secret_name not in self._cache_timestamps:
            return False
        
        current_time = time.time()
        cache_age = current_time - self._cache_timestamps[secret_name]
        return cache_age < self.config.cache_ttl_seconds
    
    def get_secret(self, secret_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Retrieve a secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret to retrieve
            use_cache: Whether to use cached value if available
            
        Returns:
            Dictionary containing the secret data
            
        Raises:
            ClientError: If the secret cannot be retrieved
            ValueError: If the secret data is invalid JSON
        """
        full_secret_name = f"{self.config.secrets_prefix}-{secret_name}"
        
        # Check cache first
        if use_cache and self._is_cache_valid(full_secret_name):
            logger.debug(f"Retrieved secret '{secret_name}' from cache")
            return self._cache[full_secret_name]
        
        try:
            logger.debug(f"Retrieving secret '{secret_name}' from AWS Secrets Manager")
            response = self.client.get_secret_value(SecretId=full_secret_name)
            
            # Parse the secret data
            if 'SecretString' in response:
                secret_data = json.loads(response['SecretString'])
            else:
                # Handle binary secrets
                secret_data = json.loads(response['SecretBinary'].decode('utf-8'))
            
            # Cache the result
            if use_cache:
                self._cache[full_secret_name] = secret_data
                self._cache_timestamps[full_secret_name] = time.time()
                logger.debug(f"Cached secret '{secret_name}' for {self.config.cache_ttl_seconds} seconds")
            
            logger.info(f"Successfully retrieved secret '{secret_name}'")
            return secret_data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.error(f"Secret '{secret_name}' not found in AWS Secrets Manager")
                raise
            elif error_code == 'AccessDeniedException':
                logger.error(f"Access denied to secret '{secret_name}'. Check IAM permissions.")
                raise
            else:
                logger.error(f"Failed to retrieve secret '{secret_name}': {e}")
                raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in secret '{secret_name}': {e}")
            raise ValueError(f"Secret '{secret_name}' contains invalid JSON data")
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret '{secret_name}': {e}")
            raise
    
    def get_api_key(self, service_name: str) -> str:
        """
        Retrieve an API key for a specific service.
        
        Args:
            service_name: Name of the service (e.g., 'openai', 'pdl', 'clearbit')
            
        Returns:
            API key string
            
        Raises:
            KeyError: If the API key is not found in the secret
        """
        try:
            secret_data = self.get_secret(f"api-keys")
            api_key = secret_data.get(f"{service_name}_api_key")
            
            if not api_key:
                raise KeyError(f"API key for service '{service_name}' not found in secret")
            
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for service '{service_name}': {e}")
            raise
    
    def get_database_credentials(self) -> Dict[str, str]:
        """
        Retrieve database credentials.
        
        Returns:
            Dictionary containing database connection parameters
            
        Raises:
            KeyError: If required database credentials are missing
        """
        try:
            secret_data = self.get_secret("database-credentials")
            required_keys = ['host', 'port', 'database', 'username', 'password']
            
            missing_keys = [key for key in required_keys if key not in secret_data]
            if missing_keys:
                raise KeyError(f"Missing required database credentials: {missing_keys}")
            
            return {
                'host': secret_data['host'],
                'port': secret_data['port'],
                'database': secret_data['database'],
                'username': secret_data['username'],
                'password': secret_data['password']
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve database credentials: {e}")
            raise
    
    def get_redis_credentials(self) -> Dict[str, str]:
        """
        Retrieve Redis credentials.
        
        Returns:
            Dictionary containing Redis connection parameters
        """
        try:
            secret_data = self.get_secret("redis-credentials")
            return {
                'host': secret_data.get('host', 'localhost'),
                'port': secret_data.get('port', '6379'),
                'password': secret_data.get('password'),
                'db': secret_data.get('db', '0')
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve Redis credentials: {e}")
            raise
    
    def clear_cache(self, secret_name: Optional[str] = None):
        """
        Clear the secret cache.
        
        Args:
            secret_name: Specific secret to clear from cache. If None, clears all cache.
        """
        if secret_name:
            full_secret_name = f"{self.config.secrets_prefix}-{secret_name}"
            self._cache.pop(full_secret_name, None)
            self._cache_timestamps.pop(full_secret_name, None)
            logger.debug(f"Cleared cache for secret '{secret_name}'")
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.debug("Cleared all secret cache")
    
    def list_secrets(self) -> list:
        """
        List all secrets with the configured prefix.
        
        Returns:
            List of secret names
        """
        try:
            response = self.client.list_secrets()
            secrets = []
            
            for secret in response.get('SecretList', []):
                secret_name = secret['Name']
                if secret_name.startswith(self.config.secrets_prefix):
                    # Remove prefix from name
                    clean_name = secret_name[len(self.config.secrets_prefix) + 1:]
                    secrets.append(clean_name)
            
            return secrets
            
        except ClientError as e:
            logger.error(f"Failed to list secrets: {e}")
            raise
    
    def create_secret(self, secret_name: str, secret_data: Dict[str, Any], description: str = "", custom_tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new secret in AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret
            secret_data: Dictionary containing the secret data
            description: Optional description for the secret
            custom_tags: Optional dictionary of custom tags to add
            
        Returns:
            ARN of the created secret
        """
        full_secret_name = f"{self.config.secrets_prefix}-{secret_name}"
        
        # Build tags list with defaults and custom tags
        tags = [
            {
                'Key': 'Environment',
                'Value': self.config.environment
            },
            {
                'Key': 'Application',
                'Value': self.config.application_name
            }
        ]
        
        # Add custom tags if provided
        if custom_tags:
            for key, value in custom_tags.items():
                tags.append({
                    'Key': key,
                    'Value': value
                })
        
        try:
            response = self.client.create_secret(
                Name=full_secret_name,
                Description=description,
                SecretString=json.dumps(secret_data),
                Tags=tags
            )
            
            logger.info(f"Successfully created secret '{secret_name}' with tags: {[tag['Key'] for tag in tags]}")
            return response['ARN']
            
        except ClientError as e:
            logger.error(f"Failed to create secret '{secret_name}': {e}")
            raise
    
    def update_secret(self, secret_name: str, secret_data: Dict[str, Any]) -> str:
        """
        Update an existing secret in AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret to update
            secret_data: New secret data
            
        Returns:
            ARN of the updated secret
        """
        full_secret_name = f"{self.config.secrets_prefix}-{secret_name}"
        
        try:
            response = self.client.update_secret(
                SecretId=full_secret_name,
                SecretString=json.dumps(secret_data)
            )
            
            # Clear cache for this secret
            self.clear_cache(secret_name)
            
            logger.info(f"Successfully updated secret '{secret_name}'")
            return response['ARN']
            
        except ClientError as e:
            logger.error(f"Failed to update secret '{secret_name}': {e}")
            raise


# Global secrets manager instance
@lru_cache()
def get_secrets_manager() -> SecretsManager:
    """Get a cached instance of the Secrets Manager."""
    return SecretsManager()


# Convenience functions for common operations
def get_openai_api_key() -> str:
    """Get OpenAI API key from secrets manager."""
    return get_secrets_manager().get_api_key('openai')


def get_pdl_api_key() -> str:
    """Get PDL API key from secrets manager."""
    return get_secrets_manager().get_api_key('pdl')


def get_clearbit_api_key() -> str:
    """Get Clearbit API key from secrets manager."""
    return get_secrets_manager().get_api_key('clearbit')


def get_database_config() -> Dict[str, str]:
    """Get database configuration from secrets manager."""
    return get_secrets_manager().get_database_credentials()


def get_redis_config() -> Dict[str, str]:
    """Get Redis configuration from secrets manager."""
    return get_secrets_manager().get_redis_credentials() 