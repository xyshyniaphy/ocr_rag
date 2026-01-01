"""
Logging Configuration
Structured logging with JSON output for production
"""

import logging
import sys
from typing import Any

from loguru import logger as loguru_logger

from backend.core.config import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru"""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Setup application logging"""

    # Remove default handler
    loguru_logger.remove()

    # Configure console output
    if settings.DEBUG:
        # Development: Pretty colored output
        loguru_logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",
            colorize=True,
        )
    else:
        # Production: Structured JSON output
        loguru_logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=settings.LOG_LEVEL,
            serialize=True,  # JSON output
        )

    # Add file output
    loguru_logger.add(
        "/app/logs/app.log",
        rotation="100 MB",
        retention="7 days",
        compression="zip",
        level=settings.LOG_LEVEL,
        serialize=True,
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set log levels for third-party libraries
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]
    logging.getLogger("fastapi").handlers = [InterceptHandler()]
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def get_logger(name: str) -> Any:
    """Get a logger instance"""
    return loguru_logger.bind(name=name)
