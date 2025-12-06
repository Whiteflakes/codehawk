"""Unit tests for configuration."""

import pytest
from codehawk.config import Settings


def test_settings_initialization():
    """Test settings initialization with defaults."""
    settings = Settings()
    
    assert settings.db_host == "localhost"
    assert settings.db_port == 5432
    assert settings.db_name == "codehawk"
    assert settings.embedding_dimension == 384
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50


def test_database_url():
    """Test database URL construction."""
    settings = Settings(
        db_host="testhost",
        db_port=5433,
        db_name="testdb",
        db_user="testuser",
        db_password="testpass",
    )
    
    url = settings.database_url
    assert "testhost" in url
    assert "5433" in url
    assert "testdb" in url
    assert "testuser" in url
    assert "testpass" in url


def test_supported_languages():
    """Test supported languages list."""
    settings = Settings()
    
    assert "python" in settings.supported_languages
    assert "javascript" in settings.supported_languages
    assert "typescript" in settings.supported_languages
