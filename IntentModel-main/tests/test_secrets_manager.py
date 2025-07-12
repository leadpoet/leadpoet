"""
Tests for AWS Secrets Manager integration.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError, NoCredentialsError

from app.core.secrets_manager import (
    SecretsManager,
    SecretsManagerConfig,
    get_secrets_manager,
    get_openai_api_key,
    get_pdl_api_key,
    get_clearbit_api_key,
    get_database_config,
    get_redis_config
)


class TestSecretsManagerConfig:
    """Test SecretsManagerConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SecretsManagerConfig()
        assert config.aws_region == "us-east-1"
        assert config.secrets_prefix == "leadpoet"
        assert config.cache_ttl_seconds == 300
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SecretsManagerConfig(
            aws_region="us-west-2",
            secrets_prefix="test",
            cache_ttl_seconds=600
        )
        assert config.aws_region == "us-west-2"
        assert config.secrets_prefix == "test"
        assert config.cache_ttl_seconds == 600


class TestSecretsManager:
    """Test SecretsManager class."""
    
    @pytest.fixture
    def mock_boto3_client(self):
        """Mock boto3 client for testing."""
        with patch('boto3.client') as mock_client:
            yield mock_client
    
    @pytest.fixture
    def secrets_manager(self, mock_boto3_client):
        """Create a SecretsManager instance with mocked client."""
        config = SecretsManagerConfig(secrets_prefix="test")
        return SecretsManager(config)
    
    def test_client_initialization_success(self, mock_boto3_client):
        """Test successful client initialization."""
        config = SecretsManagerConfig(aws_region="us-west-2")
        manager = SecretsManager(config)
        
        # Access client property to trigger initialization
        _ = manager.client
        
        mock_boto3_client.assert_called_once_with(
            'secretsmanager',
            region_name='us-west-2'
        )
    
    def test_client_initialization_no_credentials(self, mock_boto3_client):
        """Test client initialization with no AWS credentials."""
        mock_boto3_client.side_effect = NoCredentialsError()
        config = SecretsManagerConfig()
        manager = SecretsManager(config)
        
        with pytest.raises(NoCredentialsError):
            _ = manager.client
    
    def test_get_secret_success(self, secrets_manager, mock_boto3_client):
        """Test successful secret retrieval."""
        mock_response = {
            'SecretString': json.dumps({
                'api_key': 'test-api-key',
                'username': 'test-user'
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        result = secrets_manager.get_secret('test-secret')
        
        assert result['api_key'] == 'test-api-key'
        assert result['username'] == 'test-user'
        mock_boto3_client.return_value.get_secret_value.assert_called_once_with(
            SecretId='test-test-secret'
        )
    
    def test_get_secret_binary(self, secrets_manager, mock_boto3_client):
        """Test secret retrieval with binary data."""
        secret_data = {'api_key': 'test-api-key'}
        mock_response = {
            'SecretBinary': json.dumps(secret_data).encode('utf-8')
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        result = secrets_manager.get_secret('test-secret')
        
        assert result['api_key'] == 'test-api-key'
    
    def test_get_secret_not_found(self, secrets_manager, mock_boto3_client):
        """Test secret retrieval when secret doesn't exist."""
        error_response = {
            'Error': {
                'Code': 'ResourceNotFoundException',
                'Message': 'Secret not found'
            }
        }
        mock_boto3_client.return_value.get_secret_value.side_effect = ClientError(
            error_response, 'GetSecretValue'
        )
        
        with pytest.raises(ClientError):
            secrets_manager.get_secret('nonexistent-secret')
    
    def test_get_secret_access_denied(self, secrets_manager, mock_boto3_client):
        """Test secret retrieval with access denied."""
        error_response = {
            'Error': {
                'Code': 'AccessDeniedException',
                'Message': 'Access denied'
            }
        }
        mock_boto3_client.return_value.get_secret_value.side_effect = ClientError(
            error_response, 'GetSecretValue'
        )
        
        with pytest.raises(ClientError):
            secrets_manager.get_secret('restricted-secret')
    
    def test_get_secret_invalid_json(self, secrets_manager, mock_boto3_client):
        """Test secret retrieval with invalid JSON."""
        mock_response = {
            'SecretString': 'invalid-json{'
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        with pytest.raises(ValueError, match="contains invalid JSON data"):
            secrets_manager.get_secret('invalid-secret')
    
    def test_get_secret_caching(self, secrets_manager, mock_boto3_client):
        """Test secret caching functionality."""
        mock_response = {
            'SecretString': json.dumps({'key': 'value'})
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        # First call should hit AWS
        result1 = secrets_manager.get_secret('test-secret')
        assert result1['key'] == 'value'
        
        # Second call should use cache
        result2 = secrets_manager.get_secret('test-secret')
        assert result2['key'] == 'value'
        
        # Should only call AWS once
        assert mock_boto3_client.return_value.get_secret_value.call_count == 1
    
    def test_get_secret_cache_bypass(self, secrets_manager, mock_boto3_client):
        """Test secret retrieval bypassing cache."""
        mock_response = {
            'SecretString': json.dumps({'key': 'value'})
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        # First call with cache
        secrets_manager.get_secret('test-secret')
        
        # Second call without cache
        secrets_manager.get_secret('test-secret', use_cache=False)
        
        # Should call AWS twice
        assert mock_boto3_client.return_value.get_secret_value.call_count == 2
    
    def test_get_api_key_success(self, secrets_manager, mock_boto3_client):
        """Test successful API key retrieval."""
        mock_response = {
            'SecretString': json.dumps({
                'openai_api_key': 'sk-test-key',
                'pdl_api_key': 'pdl-test-key'
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        result = secrets_manager.get_api_key('openai')
        assert result == 'sk-test-key'
    
    def test_get_api_key_not_found(self, secrets_manager, mock_boto3_client):
        """Test API key retrieval when key doesn't exist."""
        mock_response = {
            'SecretString': json.dumps({
                'openai_api_key': 'sk-test-key'
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        with pytest.raises(KeyError, match="API key for service 'pdl' not found"):
            secrets_manager.get_api_key('pdl')
    
    def test_get_database_credentials_success(self, secrets_manager, mock_boto3_client):
        """Test successful database credentials retrieval."""
        mock_response = {
            'SecretString': json.dumps({
                'host': 'db.example.com',
                'port': '5432',
                'database': 'leadpoet',
                'username': 'dbuser',
                'password': 'dbpass'
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        result = secrets_manager.get_database_credentials()
        
        assert result['host'] == 'db.example.com'
        assert result['port'] == '5432'
        assert result['database'] == 'leadpoet'
        assert result['username'] == 'dbuser'
        assert result['password'] == 'dbpass'
    
    def test_get_database_credentials_missing_keys(self, secrets_manager, mock_boto3_client):
        """Test database credentials retrieval with missing required keys."""
        mock_response = {
            'SecretString': json.dumps({
                'host': 'db.example.com',
                'port': '5432'
                # Missing database, username, password
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        with pytest.raises(KeyError, match="Missing required database credentials"):
            secrets_manager.get_database_credentials()
    
    def test_get_redis_credentials_success(self, secrets_manager, mock_boto3_client):
        """Test successful Redis credentials retrieval."""
        mock_response = {
            'SecretString': json.dumps({
                'host': 'redis.example.com',
                'port': '6379',
                'password': 'redispass',
                'db': '1'
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        result = secrets_manager.get_redis_credentials()
        
        assert result['host'] == 'redis.example.com'
        assert result['port'] == '6379'
        assert result['password'] == 'redispass'
        assert result['db'] == '1'
    
    def test_get_redis_credentials_defaults(self, secrets_manager, mock_boto3_client):
        """Test Redis credentials retrieval with default values."""
        mock_response = {
            'SecretString': json.dumps({
                'host': 'redis.example.com'
                # Missing port, password, db
            })
        }
        mock_boto3_client.return_value.get_secret_value.return_value = mock_response
        
        result = secrets_manager.get_redis_credentials()
        
        assert result['host'] == 'redis.example.com'
        assert result['port'] == '6379'  # Default
        assert result['password'] is None
        assert result['db'] == '0'  # Default
    
    def test_clear_cache_specific(self, secrets_manager):
        """Test clearing specific secret from cache."""
        # Add something to cache
        secrets_manager._cache['test-secret'] = {'key': 'value'}
        secrets_manager._cache_timestamps['test-secret'] = 1234567890
        
        # Clear specific secret
        secrets_manager.clear_cache('test-secret')
        
        assert 'test-secret' not in secrets_manager._cache
        assert 'test-secret' not in secrets_manager._cache_timestamps
    
    def test_clear_cache_all(self, secrets_manager):
        """Test clearing all secrets from cache."""
        # Add multiple items to cache
        secrets_manager._cache['secret1'] = {'key1': 'value1'}
        secrets_manager._cache['secret2'] = {'key2': 'value2'}
        secrets_manager._cache_timestamps['secret1'] = 1234567890
        secrets_manager._cache_timestamps['secret2'] = 1234567891
        
        # Clear all cache
        secrets_manager.clear_cache()
        
        assert len(secrets_manager._cache) == 0
        assert len(secrets_manager._cache_timestamps) == 0
    
    def test_list_secrets_success(self, secrets_manager, mock_boto3_client):
        """Test successful listing of secrets."""
        mock_response = {
            'SecretList': [
                {'Name': 'test-secret1'},
                {'Name': 'test-secret2'},
                {'Name': 'other-secret'},  # Should be filtered out
                {'Name': 'test-secret3'}
            ]
        }
        mock_boto3_client.return_value.list_secrets.return_value = mock_response
        
        result = secrets_manager.list_secrets()
        
        expected = ['secret1', 'secret2', 'secret3']
        assert result == expected
    
    def test_create_secret_success(self, secrets_manager, mock_boto3_client):
        """Test successful secret creation."""
        mock_response = {'ARN': 'arn:aws:secretsmanager:region:account:secret:test-secret'}
        mock_boto3_client.return_value.create_secret.return_value = mock_response
        
        secret_data = {'api_key': 'test-key'}
        result = secrets_manager.create_secret('new-secret', secret_data, 'Test secret')
        
        assert result == 'arn:aws:secretsmanager:region:account:secret:test-secret'
        mock_boto3_client.return_value.create_secret.assert_called_once()
    
    def test_update_secret_success(self, secrets_manager, mock_boto3_client):
        """Test successful secret update."""
        mock_response = {'ARN': 'arn:aws:secretsmanager:region:account:secret:test-secret'}
        mock_boto3_client.return_value.update_secret.return_value = mock_response
        
        # Add to cache first
        secrets_manager._cache['test-existing-secret'] = {'old': 'data'}
        secrets_manager._cache_timestamps['test-existing-secret'] = 1234567890
        
        secret_data = {'api_key': 'new-key'}
        result = secrets_manager.update_secret('existing-secret', secret_data)
        
        assert result == 'arn:aws:secretsmanager:region:account:secret:test-secret'
        # Cache should be cleared
        assert 'test-existing-secret' not in secrets_manager._cache


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('app.core.secrets_manager.get_secrets_manager')
    def test_get_openai_api_key(self, mock_get_manager):
        """Test get_openai_api_key convenience function."""
        mock_manager = Mock()
        mock_manager.get_api_key.return_value = 'sk-test-key'
        mock_get_manager.return_value = mock_manager
        
        result = get_openai_api_key()
        
        assert result == 'sk-test-key'
        mock_manager.get_api_key.assert_called_once_with('openai')
    
    @patch('app.core.secrets_manager.get_secrets_manager')
    def test_get_pdl_api_key(self, mock_get_manager):
        """Test get_pdl_api_key convenience function."""
        mock_manager = Mock()
        mock_manager.get_api_key.return_value = 'pdl-test-key'
        mock_get_manager.return_value = mock_manager
        
        result = get_pdl_api_key()
        
        assert result == 'pdl-test-key'
        mock_manager.get_api_key.assert_called_once_with('pdl')
    
    @patch('app.core.secrets_manager.get_secrets_manager')
    def test_get_clearbit_api_key(self, mock_get_manager):
        """Test get_clearbit_api_key convenience function."""
        mock_manager = Mock()
        mock_manager.get_api_key.return_value = 'clearbit-test-key'
        mock_get_manager.return_value = mock_manager
        
        result = get_clearbit_api_key()
        
        assert result == 'clearbit-test-key'
        mock_manager.get_api_key.assert_called_once_with('clearbit')
    
    @patch('app.core.secrets_manager.get_secrets_manager')
    def test_get_database_config(self, mock_get_manager):
        """Test get_database_config convenience function."""
        mock_manager = Mock()
        mock_manager.get_database_credentials.return_value = {
            'host': 'db.example.com',
            'port': '5432',
            'database': 'leadpoet',
            'username': 'dbuser',
            'password': 'dbpass'
        }
        mock_get_manager.return_value = mock_manager
        
        result = get_database_config()
        
        assert result['host'] == 'db.example.com'
        mock_manager.get_database_credentials.assert_called_once()
    
    @patch('app.core.secrets_manager.get_secrets_manager')
    def test_get_redis_config(self, mock_get_manager):
        """Test get_redis_config convenience function."""
        mock_manager = Mock()
        mock_manager.get_redis_credentials.return_value = {
            'host': 'redis.example.com',
            'port': '6379',
            'password': 'redispass',
            'db': '0'
        }
        mock_get_manager.return_value = mock_manager
        
        result = get_redis_config()
        
        assert result['host'] == 'redis.example.com'
        mock_manager.get_redis_credentials.assert_called_once()


class TestGetSecretsManager:
    """Test get_secrets_manager function."""
    
    def test_get_secrets_manager_caching(self):
        """Test that get_secrets_manager returns cached instance."""
        # Clear any existing cache
        get_secrets_manager.cache_clear()
        
        # First call
        manager1 = get_secrets_manager()
        
        # Second call
        manager2 = get_secrets_manager()
        
        # Should be the same instance
        assert manager1 is manager2 