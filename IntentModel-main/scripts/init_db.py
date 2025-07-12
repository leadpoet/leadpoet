#!/usr/bin/env python3
"""
Database initialization script for Leadpoet Intent Model v1.1
Initializes the database with required extensions and runs migrations.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import init_db, test_db_connection
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Initialize the database."""
    try:
        logger.info("Initializing Leadpoet database...")
        
        # Initialize database extensions
        await init_db()
        logger.info("✅ Database extensions initialized")
        
        # Test database connection
        if await test_db_connection():
            logger.info("✅ Database connection test successful")
        else:
            logger.error("❌ Database connection test failed")
            sys.exit(1)
        
        # Run Alembic migrations
        logger.info("Running database migrations...")
        import subprocess
        
        try:
            # Run alembic upgrade
            result = subprocess.run(
                ["alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300,  # 5 minute timeout
                env=os.environ.copy()  # Explicit environment passing
            )

            if result.returncode == 0:
                logger.info("✅ Database migrations completed successfully")
                if result.stdout.strip():
                    logger.info(f"Migration output: {result.stdout}")
            else:
                logger.error("❌ Database migrations failed")
        except FileNotFoundError:
            logger.error("❌ Alembic not found. Please install it: pip install alembic")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.error("❌ Database migrations timed out")
            sys.exit(1)                sys.exit(1)        except FileNotFoundError:
            logger.error("❌ Alembic not found. Please install it: pip install alembic")
            sys.exit(1)
        
        logger.info("🎉 Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 