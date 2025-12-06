"""Utility functions for CodeHawk."""

import logging
from typing import List, Set
from pathlib import Path

logger = logging.getLogger(__name__)


def get_supported_extensions() -> Set[str]:
    """Get set of supported file extensions."""
    return {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs"}


def is_code_file(file_path: Path) -> bool:
    """
    Check if a file is a supported code file.

    Args:
        file_path: Path to file

    Returns:
        True if file is a supported code file
    """
    return file_path.suffix in get_supported_extensions()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
