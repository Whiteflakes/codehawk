"""Unit tests for tree-sitter parser."""

import importlib.util
import importlib
from pathlib import Path

import pytest

from codehawk.parser import TreeSitterParser


def test_parser_initialization():
    """Test parser initialization."""
    parser = TreeSitterParser()
    assert parser is not None
    assert isinstance(parser.parsers, dict)
    assert isinstance(parser.languages, dict)


def test_detect_language():
    """Test language detection from file extension."""
    parser = TreeSitterParser()

    assert parser._detect_language(Path("test.py")) == "python"
    assert parser._detect_language(Path("test.js")) == "javascript"
    assert parser._detect_language(Path("test.ts")) == "typescript"
    assert parser._detect_language(Path("test.java")) == "java"
    assert parser._detect_language(Path("test.go")) == "go"
    assert parser._detect_language(Path("test.rs")) == "rust"
    assert parser._detect_language(Path("test.txt")) == "unknown"


def test_parsers_registered_for_supported_extensions():
    """Ensure available language parsers are registered for supported extensions."""
    parser = TreeSitterParser()

    language_modules = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "java": "tree_sitter_java",
        "go": "tree_sitter_go",
        "rust": "tree_sitter_rust",
    }

    for extension, language in parser.EXTENSION_LANGUAGE_MAP.items():
        assert parser._detect_language(Path(f"file{extension}")) == language

        module_name = language_modules.get(language)
        if module_name is None:
            continue

        try:
            module_spec = importlib.util.find_spec(module_name)
            module_available = module_spec is not None
            module = importlib.import_module(module_name) if module_available else None
            module_available = module_available and hasattr(module, "language")
        except (ImportError, AttributeError):
            module_available = False

        if module_available:
            assert language in parser.parsers
            assert language in parser.languages
        else:
            assert language not in parser.parsers


def test_parse_python_code():
    """Test parsing Python code."""
    parser = TreeSitterParser()
    
    if "python" not in parser.parsers:
        pytest.skip("Python parser not available")
    
    code = """
def hello_world():
    print("Hello, World!")
    return True
"""
    
    tree = parser.parse_code(code, "python")
    assert tree is not None
    assert tree.root_node is not None


def test_parse_invalid_language():
    """Test parsing with invalid language."""
    parser = TreeSitterParser()
    
    code = "some code"
    tree = parser.parse_code(code, "invalid_language")
    
    assert tree is None
