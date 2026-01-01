#!/usr/bin/env python3
"""
Database Initialization Script
Initialize PostgreSQL database with tables
"""

import asyncio
import sys

from backend.db.session import init_db
from backend.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def main():
    """Main initialization function"""
    logger.info("Initializing database...")

    try:
        await init_db()
        logger.info("Database initialized successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
