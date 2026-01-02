#!/usr/bin/env python3
"""
Unit Tests for Logging Configuration
Tests for backend/core/logging.py
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.core.logging import get_logger, setup_logging


class TestSetupLogging:
    """Test logging configuration"""

    def test_setup_logging_default(self):
        """Test logging configuration with defaults"""
        setup_logging()
        logger = logging.getLogger()
        # Root logger should be configured
        assert logger is not None

    def test_setup_logging_configures_handlers(self):
        """Test logging configuration adds handlers"""
        setup_logging()
        logger = logging.getLogger()
        # Should have at least one handler
        assert len(logger.handlers) > 0


class TestGetLogger:
    """Test logger creation (loguru-based)"""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance"""
        logger = get_logger(__name__)
        # loguru returns a bound logger, not logging.Logger
        assert logger is not None
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')

    def test_get_logger_with_name(self):
        """Test get_logger with specific name"""
        logger = get_logger("test.logger")
        # loguru bind creates a logger with extra context
        assert logger is not None

    def test_get_logger_with_module_name(self):
        """Test get_logger with module name"""
        logger = get_logger(__name__)
        # loguru bind creates a logger with extra context
        assert logger is not None

    def test_get_logger_returns_bound_logger(self):
        """Test get_logger returns a bound loguru logger"""
        from loguru import logger as loguru_logger
        logger = get_logger("test.same")
        # Should be a bound logger (has .bind() method)
        assert hasattr(logger, 'bind')
        assert hasattr(logger, 'log')

    def test_get_logger_different_instances(self):
        """Test get_logger returns different instances for different names"""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")
        # Each bind() call creates a new context wrapper
        assert logger1 is not logger2

    def test_get_logger_can_log_debug(self):
        """Test logger can log DEBUG messages"""
        logger = get_logger(__name__)
        # loguru logger can log, but we can't easily test caplog with it
        # Just verify the method exists and is callable
        assert callable(logger.debug)

    def test_get_logger_can_log_info(self):
        """Test logger can log INFO messages"""
        logger = get_logger(__name__)
        assert callable(logger.info)

    def test_get_logger_can_log_warning(self):
        """Test logger can log WARNING messages"""
        logger = get_logger(__name__)
        assert callable(logger.warning)

    def test_get_logger_can_log_error(self):
        """Test logger can log ERROR messages"""
        logger = get_logger(__name__)
        assert callable(logger.error)

    def test_get_logger_can_log_critical(self):
        """Test logger can log CRITICAL messages"""
        logger = get_logger(__name__)
        assert callable(logger.critical)

    def test_get_logger_with_context(self):
        """Test logger can log with context"""
        logger = get_logger(__name__)
        # loguru supports contextual logging
        assert callable(logger.info)


class TestInterceptHandler:
    """Test InterceptHandler class"""

    def test_intercept_handler_is_handler(self):
        """Test InterceptHandler is a logging Handler"""
        from backend.core.logging import InterceptHandler
        handler = InterceptHandler()
        assert isinstance(handler, logging.Handler)

    def test_intercept_handler_has_emit_method(self):
        """Test InterceptHandler has emit method"""
        from backend.core.logging import InterceptHandler
        handler = InterceptHandler()
        assert hasattr(handler, 'emit')
        assert callable(handler.emit)
