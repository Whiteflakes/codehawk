"""Unit tests for code chunker."""

import pytest
from pathlib import Path
from codehawk.chunker import CodeChunker, CodeChunk


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = CodeChunker(chunk_size=512, overlap=50)
    assert chunker.chunk_size == 512
    assert chunker.overlap == 50


def test_chunk_by_lines():
    """Test line-based chunking."""
    chunker = CodeChunker(chunk_size=60, overlap=10)
    
    source_code = "\n".join([f"line {i}" for i in range(20)])
    
    chunks = chunker._chunk_by_lines(
        file_path=Path("test.py"),
        language="python",
        source_code=source_code,
    )
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, CodeChunk)
        assert chunk.language == "python"
        assert chunk.file_path == "test.py"
        assert chunk.chunk_type == "block"


def test_chunk_small_file():
    """Test chunking a small file."""
    chunker = CodeChunker(chunk_size=512, overlap=50)
    
    source_code = """
def hello():
    print("Hello")

def world():
    print("World")
"""
    
    chunks = chunker._chunk_by_lines(
        file_path=Path("test.py"),
        language="python",
        source_code=source_code,
    )
    
    assert len(chunks) >= 1
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)


def test_is_chunkable_node():
    """Test node chunkability detection."""
    chunker = CodeChunker()
    
    # Mock node with type attribute
    class MockNode:
        def __init__(self, node_type):
            self.type = node_type
    
    # Python
    assert chunker._is_chunkable_node(MockNode("function_definition"), "python")
    assert chunker._is_chunkable_node(MockNode("class_definition"), "python")
    assert not chunker._is_chunkable_node(MockNode("import_statement"), "python")
    
    # JavaScript
    assert chunker._is_chunkable_node(MockNode("function_declaration"), "javascript")
    assert chunker._is_chunkable_node(MockNode("class_declaration"), "javascript")
    assert not chunker._is_chunkable_node(MockNode("variable_declaration"), "javascript")
