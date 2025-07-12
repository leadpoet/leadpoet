"""
Database connection module for PostgreSQL with TimescaleDB.
Provides async database connections and session management.
"""

import asyncio
from typing import AsyncGenerator, Generator
import logging

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import text, create_engine

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create async engine
if not settings.database_url:
    raise ValueError("Database URL not configured")

# Ensure URL is in expected format
if not settings.database_url.startswith(("postgresql://", "postgresql+asyncpg://")):
    raise ValueError(f"Invalid database URL format: {settings.database_url}")

# Async engine for async operations
async_engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
        if settings.database_url.startswith("postgresql://")
        else settings.database_url,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Sync engine for sync operations (services that need it)
sync_engine = create_engine(
    settings.database_url,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Create session factories
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Sync session factory for services
SessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False
)

# Base class for models
from sqlalchemy.orm import DeclarativeBase

# Base class for models
class Base(DeclarativeBase):
    pass

def get_sync_db() -> Generator[Session, None, None]:
    """Dependency to get synchronous database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database with required extensions and tables."""
    try:
        async with async_engine.begin() as conn:
            # Enable TimescaleDB extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            
            # Enable pgvector extension for ANN search
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            logger.info("Database extensions initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database extensions: {e}")
        logger.error("Ensure the database user has SUPERUSER or CREATE EXTENSION privileges")
        raise

async def test_db_connection():
    """Test database connection."""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            await session.commit()
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def close_db():
    """Close database connections."""
    await async_engine.dispose()
    sync_engine.dispose()
    logger.info("Database connections closed") 