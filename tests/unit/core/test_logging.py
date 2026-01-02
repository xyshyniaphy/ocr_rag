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
    """Test logger creation"""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance"""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test get_logger with specific name"""
        logger = get_logger("test.logger")
        assert logger.name == "test.logger"

    def test_get_logger_with_module_name(self):
        """Test get_logger with module name"""
        logger = get_logger(__name__)
        assert logger.name == __name__

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same instance for same name"""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_get_logger_different_instances(self):
        """Test get_logger returns different instances for different names"""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")
        assert logger1 is not logger2

    def test_get_logger_can_log_debug(self, caplog):
        """Test logger can log DEBUG messages"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.DEBUG):
            logger.debug("Test debug message")
            assert "Test debug message" in caplog.text

    def test_get_logger_can_log_info(self, caplog):
        """Test logger can log INFO messages"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("Test info message")
            assert "Test info message" in caplog.text

    def test_get_logger_can_log_warning(self, caplog):
        """Test logger can log WARNING messages"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.WARNING):
            logger.warning("Test warning message")
            assert "Test warning message" in caplog.text

    def test_get_logger_can_log_error(self, caplog):
        """Test logger can log ERROR messages"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.ERROR):
            logger.error("Test error message")
            assert "Test error message" in caplog.text

    def test_get_logger_can_log_critical(self, caplog):
        """Test logger can log CRITICAL messages"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.CRITICAL):
            logger.critical("Test critical message")
            assert "Test critical message" in caplog.text

    def test_get_logger_with_context(self, caplog):
        """Test logger can log with context"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("User %s performed action", "user123")
            assert "User user123 performed action" in caplog.text


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
