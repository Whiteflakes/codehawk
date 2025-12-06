"""Configuration management for CodeHawk."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="CODEHAWK_", env_file=".env")

    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "codehawk"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_offline_mode: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # MCP server settings
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001

    # Indexing settings
    max_file_size: int = 1_000_000  # 1MB
    supported_languages: list[str] = [
        "python",
        "javascript",
        "typescript",
        "java",
        "go",
        "rust",
    ]

    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Global settings instance
settings = Settings()
