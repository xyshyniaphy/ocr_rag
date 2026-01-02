#!/usr/bin/env python3
"""
Unit Tests for Security Utilities
Tests for backend/core/security.py
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import time

from backend.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_access_token,
    verify_refresh_token,
)
from backend.core.exceptions import AuthenticationException


class TestPasswordHashing:
    """Test password hashing and verification"""

    def test_get_password_hash_returns_hash(self):
        """Test password hashing returns a hash"""
        password = "test_password_123"
        hashed = get_password_hash(password)

        assert isinstance(hashed, str)
        assert len(hashed) > 0
        assert hashed != password  # Hash should be different from password

    def test_get_password_hash_is_consistent(self):
        """Test password hashing is consistent for same password"""
        password = "test_password_123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # bcrypt generates different hashes each time (with different salts)
        # but they should both verify correctly
        assert hash1 != hash2  # Different hashes due to salt
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)

    def test_verify_password_correct_password(self):
        """Test verifying correct password returns True"""
        password = "test_password_123"
        hashed = get_password_hash(password)

        result = verify_password(password, hashed)
        assert result is True

    def test_verify_password_incorrect_password(self):
        """Test verifying incorrect password returns False"""
        password = "test_password_123"
        wrong_password = "wrong_password_456"
        hashed = get_password_hash(password)

        result = verify_password(wrong_password, hashed)
        assert result is False

    def test_verify_password_with_unicode(self):
        """Test password hashing works with unicode characters"""
        password = "パスワード123日本語"  # Japanese password
        hashed = get_password_hash(password)

        assert verify_password(password, hashed)
        assert not verify_password("different", hashed)

    def test_verify_password_with_special_characters(self):
        """Test password hashing works with special characters"""
        password = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed)


class TestAccessTokens:
    """Test JWT access token creation and verification"""

    def test_create_access_token_returns_token(self):
        """Test access token creation returns a string"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_default_expiration(self):
        """Test access token has default expiration (from environment)"""
        from backend.core.config import settings
        data = {"sub": "user123"}
        token = create_access_token(data)

        payload = decode_token(token)

        assert "exp" in payload
        assert "type" in payload
        assert payload["type"] == "access"
        assert payload["sub"] == "user123"

        # Check expiration matches environment setting (60 minutes in Docker)
        exp = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        time_diff = abs((exp - expected_exp).total_seconds())
        assert time_diff < 5  # Less than 5 seconds difference

    def test_create_access_token_custom_expiration(self):
        """Test access token with custom expiration"""
        data = {"sub": "user123"}
        expires = timedelta(hours=2)
        token = create_access_token(data, expires_delta=expires)

        payload = decode_token(token)

        # Check expiration is approximately 2 hours from now
        exp = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(hours=2)
        time_diff = abs((exp - expected_exp).total_seconds())
        assert time_diff < 5

    def test_verify_access_token_valid_token(self):
        """Test verifying a valid access token"""
        data = {"sub": "user123"}
        token = create_access_token(data)

        payload = verify_access_token(token)

        assert payload["sub"] == "user123"
        assert payload["type"] == "access"

    def test_verify_access_token_invalid_token(self):
        """Test verifying an invalid token raises exception"""
        invalid_token = "invalid.token.here"

        with pytest.raises(AuthenticationException) as exc_info:
            verify_access_token(invalid_token)

        assert "Invalid token" in str(exc_info.value)

    def test_verify_access_token_wrong_type(self):
        """Test verifying a refresh token as access token raises exception"""
        data = {"sub": "user123"}
        refresh_token = create_refresh_token(data)

        with pytest.raises(AuthenticationException) as exc_info:
            verify_access_token(refresh_token)

        assert "Invalid token type" in str(exc_info.value)

    def test_access_token_expires_after_time(self):
        """Test access token expires after configured time"""
        data = {"sub": "user123"}
        # Create token with very short expiration
        token = create_access_token(data, expires_delta=timedelta(seconds=1))

        # Should work immediately
        payload = verify_access_token(token)
        assert payload["sub"] == "user123"

        # Wait for expiration
        time.sleep(2)

        # Should fail after expiration
        with pytest.raises(AuthenticationException):
            verify_access_token(token)


class TestRefreshTokens:
    """Test JWT refresh token creation and verification"""

    def test_create_refresh_token_returns_token(self):
        """Test refresh token creation returns a string"""
        data = {"sub": "user123"}
        token = create_refresh_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token_default_expiration(self):
        """Test refresh token has default expiration (from environment)"""
        from backend.core.config import settings
        data = {"sub": "user123"}
        token = create_refresh_token(data)

        payload = decode_token(token)

        assert "exp" in payload
        assert "type" in payload
        assert payload["type"] == "refresh"
        assert payload["sub"] == "user123"

        # Check expiration matches environment setting (30 days in Docker)
        exp = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        time_diff = abs((exp - expected_exp).total_seconds())
        assert time_diff < 5

    def test_create_refresh_token_custom_expiration(self):
        """Test refresh token with custom expiration"""
        data = {"sub": "user123"}
        expires = timedelta(days=30)
        token = create_refresh_token(data, expires_delta=expires)

        payload = decode_token(token)

        # Check expiration is approximately 30 days from now
        exp = datetime.fromtimestamp(payload["exp"])
        expected_exp = datetime.utcnow() + timedelta(days=30)
        time_diff = abs((exp - expected_exp).total_seconds())
        assert time_diff < 5

    def test_verify_refresh_token_valid_token(self):
        """Test verifying a valid refresh token"""
        data = {"sub": "user123"}
        token = create_refresh_token(data)

        payload = verify_refresh_token(token)

        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"

    def test_verify_refresh_token_invalid_token(self):
        """Test verifying an invalid refresh token raises exception"""
        invalid_token = "invalid.refresh.token"

        with pytest.raises(AuthenticationException) as exc_info:
            verify_refresh_token(invalid_token)

        assert "Invalid token" in str(exc_info.value)

    def test_verify_refresh_token_wrong_type(self):
        """Test verifying an access token as refresh token raises exception"""
        data = {"sub": "user123"}
        access_token = create_access_token(data)

        with pytest.raises(AuthenticationException) as exc_info:
            verify_refresh_token(access_token)

        assert "Invalid token type" in str(exc_info.value)


class TestTokenDecoding:
    """Test JWT token decoding"""

    def test_decode_token_valid_token(self):
        """Test decoding a valid token"""
        data = {"sub": "user123", "role": "admin"}
        token = create_access_token(data)

        payload = decode_token(token)

        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"
        assert "exp" in payload

    def test_decode_token_invalid_token(self):
        """Test decoding an invalid token raises exception"""
        invalid_token = "not.a.valid.token"

        with pytest.raises(AuthenticationException) as exc_info:
            decode_token(invalid_token)

        assert "Invalid token" in str(exc_info.value)

    def test_decode_token_malformed_token(self):
        """Test decoding a malformed token raises exception"""
        malformed_tokens = [
            "",
            "only.one.part",
            "missing.signature",
            "a" * 1000,
        ]

        for token in malformed_tokens:
            with pytest.raises(AuthenticationException):
                decode_token(token)
