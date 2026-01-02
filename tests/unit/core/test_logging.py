#!/usr/bin/env python3
"""
Unit Tests for Logging Configuration
Tests for backend/core/logging.py
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.core.logging import get_logger, configure_logging


class TestConfigureLogging:
    """Test logging configuration"""

    def test_configure_logging_default(self):
        """Test logging configuration with defaults"""
        configure_logging()
        logger = logging.getLogger()
        assert logger.level == logging.INFO

    def test_configure_logging_debug(self):
        """Test logging configuration with DEBUG level"""
        configure_logging(log_level="DEBUG")
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

    def test_configure_logging_error(self):
        """Test logging configuration with ERROR level"""
        configure_logging(log_level="ERROR")
        logger = logging.getLogger()
        assert logger.level == logging.ERROR

    def test_configure_logging_with_file(self, tmp_path):
        """Test logging configuration with file output"""
        log_file = tmp_path / "test.log"
        configure_logging(log_file=str(log_file))
        assert log_file.exists()


class TestGetLogger:
    """Test logger creation"""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger instance"""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)

    def test_get_logger_name(self):
        """Test logger has correct name"""
        logger = get_logger("test.module")
        assert logger.name == "test.module"

    def test_get_logger_same_instance(self):
        """Test get_logger returns same instance for same name"""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test get_logger returns different instances for different names"""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")
        assert logger1 is not logger2
        assert logger1.name == "test.one"
        assert logger2.name == "test.two"


class TestLoggerOutput:
    """Test logger output"""

    def test_log_debug(self, caplog):
        """Test DEBUG level logging"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
        assert "Debug message" in caplog.text

    def test_log_info(self, caplog):
        """Test INFO level logging"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("Info message")
        assert "Info message" in caplog.text

    def test_log_warning(self, caplog):
        """Test WARNING level logging"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.WARNING):
            logger.warning("Warning message")
        assert "Warning message" in caplog.text

    def test_log_error(self, caplog):
        """Test ERROR level logging"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.ERROR):
            logger.error("Error message")
        assert "Error message" in caplog.text

    def test_log_critical(self, caplog):
        """Test CRITICAL level logging"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.CRITICAL):
            logger.critical("Critical message")
        assert "Critical message" in caplog.text


class TestLoggerFormat:
    """Test logger format"""

    def test_logger_has_formatter(self):
        """Test logger has formatter configured"""
        logger = get_logger(__name__)
        # Check if handlers have formatters
        assert len(logger.handlers) > 0
        for handler in logger.handlers:
            assert handler.formatter is not None

    def test_log_format_includes_timestamp(self, caplog):
        """Test log format includes timestamp"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("Test message")
        # Check if log record has timestamp
        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert hasattr(record, 'asctime')

    def test_log_format_includes_level(self, caplog):
        """Test log format includes log level"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("Test message")
        assert "INFO" in caplog.text

    def test_log_format_includes_module(self, caplog):
        """Test log format includes module name"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("Test message")
        assert "test_logging" in caplog.text


class TestContextualLogging:
    """Test contextual logging with extra fields"""

    def test_logger_with_extra_fields(self, caplog):
        """Test logger with extra context fields"""
        logger = get_logger(__name__)
        with caplog.at_level(logging.INFO):
            logger.info("Test message", extra={"user_id": "123", "request_id": "456"})
        assert "Test message" in caplog.text

    def test_logger_with_string_interpolation(self, caplog):
        """Test logger with string formatting"""
        logger = get_logger(__name__)
        user_name = "TestUser"
        with caplog.at_level(logging.INFO):
            logger.info(f"User {user_name} logged in")
        assert "User TestUser logged in" in caplog.text


class TestLoggerHandlers:
    """Test logger handlers"""

    def test_logger_has_stream_handler(self):
        """Test logger has stream handler"""
        logger = get_logger(__name__)
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )
        assert has_stream_handler

    def test_logger_handler_levels(self):
        """Test logger handler levels"""
        logger = get_logger(__name__)
        for handler in logger.handlers:
            assert handler.level in [
                logging.DEBUG, logging.INFO, logging.WARNING,
                logging.ERROR, logging.CRITICAL, logging.NOTSET
            ]


class TestLoggerInDifferentModules:
    """Test logger behavior across different modules"""

    def test_module_specific_logger(self):
        """Test creating logger for specific module"""
        logger1 = get_logger("backend.services.test")
        logger2 = get_logger("backend.api.test")
        assert logger1.name == "backend.services.test"
        assert logger2.name == "backend.api.test"

    def test_logger_hierarchy(self):
        """Test logger hierarchy"""
        parent_logger = logging.getLogger("backend")
        child_logger = logging.getLogger("backend.services")
        assert child_logger.parent == parent_logger


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
