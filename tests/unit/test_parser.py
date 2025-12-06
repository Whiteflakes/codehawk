"""Unit tests for tree-sitter parser."""

import pytest
from pathlib import Path
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
