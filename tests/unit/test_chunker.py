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

    # Java
    assert chunker._is_chunkable_node(MockNode("method_declaration"), "java")
    assert chunker._is_chunkable_node(MockNode("class_declaration"), "java")
    assert not chunker._is_chunkable_node(MockNode("field_declaration"), "java")

    # Go
    assert chunker._is_chunkable_node(MockNode("function_declaration"), "go")
    assert chunker._is_chunkable_node(MockNode("method_declaration"), "go")
    assert not chunker._is_chunkable_node(MockNode("short_var_declaration"), "go")

    # Rust
    assert chunker._is_chunkable_node(MockNode("function_item"), "rust")
    assert chunker._is_chunkable_node(MockNode("impl_item"), "rust")
    assert not chunker._is_chunkable_node(MockNode("macro_invocation"), "rust")


def test_chunk_by_structure_for_multiple_languages():
    """Ensure structured chunking works for Java, Go, and Rust nodes."""
    chunker = CodeChunker(chunk_size=512, overlap=0)

    class MockNode:
        def __init__(self, node_type, start_byte, end_byte, start_point, end_point, children=None):
            self.type = node_type
            self.start_byte = start_byte
            self.end_byte = end_byte
            self.start_point = start_point
            self.end_point = end_point
            self.children = children or []

    class MockTree:
        def __init__(self, root_node):
            self.root_node = root_node

    def build_tree(node_type: str, source: str) -> MockTree:
        node = MockNode(
            node_type=node_type,
            start_byte=0,
            end_byte=len(source.encode("utf-8")),
            start_point=(0, 0),
            end_point=(source.count("\n"), 0),
        )
        root = MockNode("program", 0, len(source.encode("utf-8")), (0, 0), (source.count("\n"), 0), [node])
        return MockTree(root)

    java_source = """public class Example {\n    void greet() {}\n}\n"""
    go_source = """package main\n\nfunc greet() {}\n"""
    rust_source = """impl Greeter {\n    fn greet() {}\n}\n"""

    java_chunks = chunker._chunk_by_structure(Path("Example.java"), build_tree("method_declaration", java_source), "java", java_source)
    go_chunks = chunker._chunk_by_structure(Path("main.go"), build_tree("function_declaration", go_source), "go", go_source)
    rust_chunks = chunker._chunk_by_structure(Path("lib.rs"), build_tree("impl_item", rust_source), "rust", rust_source)

    assert [chunk.chunk_type for chunk in java_chunks] == ["method_declaration"]
    assert [chunk.chunk_type for chunk in go_chunks] == ["function_declaration"]
    assert [chunk.chunk_type for chunk in rust_chunks] == ["impl_item"]
    assert all(chunk.language in {"java", "go", "rust"} for chunk in java_chunks + go_chunks + rust_chunks)


def test_chunk_file_falls_back_when_structure_empty():
    """Structured chunking fallback should still produce line-based chunks."""
    chunker = CodeChunker(chunk_size=120, overlap=10)

    class MockNode:
        def __init__(self, children=None):
            self.type = "program"
            self.start_byte = 0
            self.end_byte = 0
            self.start_point = (0, 0)
            self.end_point = (0, 0)
            self.children = children or []

    class MockTree:
        def __init__(self, root_node):
            self.root_node = root_node

    source_code = "line 1\nline 2\nline 3"
    empty_tree = MockTree(MockNode())

    chunks = chunker.chunk_file(Path("fallback.go"), empty_tree, "go", source_code)

    assert len(chunks) >= 1
    assert all(chunk.chunk_type == "block" for chunk in chunks)
