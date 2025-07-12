"""
Redis client module for caching and feature storage.
Provides async Redis connections with connection pooling.
"""

import json
import logging
from typing import Optional, Any, Dict, List
import asyncio

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Redis connection pool
redis_pool = ConnectionPool.from_url(
    settings.redis_url,
    max_connections=20,
    retry_on_timeout=True,
    socket_keepalive=True,
    socket_keepalive_options={
        1: 1,  # TCP_KEEPIDLE
        2: 3,  # TCP_KEEPINTVL  
        3: 5,  # TCP_KEEPCNT
    },
    socket_connect_timeout=5,
    socket_timeout=5
)

class RedisClient:
    """Async Redis client with caching utilities."""
    
    def __init__(self):
        self.redis = redis.Redis(connection_pool=redis_pool)
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        try:
            value = await self.redis.get(key)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ex: int = None) -> bool:
        """Set a key-value pair with optional expiration."""
        try:
            if ex:
                await self.redis.set(key, value, ex=ex)
            else:
                await self.redis.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set JSON value in Redis."""
        try:
            json_value = json.dumps(value)
            return await self.redis.set(key, json_value, ex=ex)
        except Exception as e:
            logger.error(f"Redis set_json error for key {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis."""
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Redis get_json error for key {key}: {e}")
            return None
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return bool(await self.redis.delete(key))
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key."""
        try:
            return bool(await self.redis.expire(key, seconds))
        except Exception as e:
            logger.error(f"Redis expire error for key {key}: {e}")
            return False
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value."""
        try:
            value = await self.redis.hget(key, field)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.error(f"Redis hget error for key {key}, field {field}: {e}")
            return None
    
    async def hset(self, key: str, field: str, value: str) -> bool:
        """Set hash field value."""
        try:
            return bool(await self.redis.hset(key, field, value))
        except Exception as e:
            logger.error(f"Redis hset error for key {key}, field {field}: {e}")
            return False
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields."""
        try:
            result = await self.redis.hgetall(key)
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in result.items()}
        except Exception as e:
            logger.error(f"Redis hgetall error for key {key}: {e}")
            return {}
    
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to sorted set."""
        try:
            return await self.redis.zadd(key, mapping)
        except Exception as e:
            logger.error(f"Redis zadd error for key {key}: {e}")
            return 0
    
    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> List:
        """Get range from sorted set."""
        try:
            return await self.redis.zrange(key, start, stop, withscores=withscores)
        except Exception as e:
            logger.error(f"Redis zrange error for key {key}: {e}")
            return []
    
    async def pipeline(self):
        """Get Redis pipeline for batch operations."""
        return self.redis.pipeline()
    
    async def close(self):
        """Close Redis connection."""
        await self.redis.close()


# Global Redis client instance
redis_client = RedisClient()


async def get_redis() -> RedisClient:
    """Dependency to get Redis client."""
    return redis_client


async def test_redis_connection() -> bool:
    """Test Redis connection."""
    try:
        await redis_client.set("test", "test", ex=10)
        value = await redis_client.get("test")
        await redis_client.delete("test")
        
        if value == "test":
            logger.info("Redis connection test successful")
            return True
        else:
            logger.error("Redis connection test failed - value mismatch")
            return False
    except Exception as e:
        logger.error(f"Redis connection test failed: {e}")
        return False


async def close_redis():
    """Close Redis connections."""
    await redis_client.close()
    await redis_pool.disconnect()
    logger.info("Redis connections closed") 