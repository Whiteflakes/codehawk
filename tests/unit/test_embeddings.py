"""Unit tests for embedding generator."""

import pytest
import numpy as np
from codehawk.embeddings import EmbeddingGenerator
from codehawk.chunker import CodeChunk


def test_embedder_initialization():
    """Test embedder initialization."""
    embedder = EmbeddingGenerator()
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.dimension == 384


def test_generate_embedding():
    """Test embedding generation."""
    embedder = EmbeddingGenerator()
    
    text = "This is a test string for embedding"
    embedding = embedder.generate_embedding(text)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedder.dimension,)
    assert embedding.dtype == np.float32


def test_generate_embeddings_batch():
    """Test batch embedding generation."""
    embedder = EmbeddingGenerator()
    
    texts = [
        "First test string",
        "Second test string",
        "Third test string",
    ]
    
    embeddings = embedder.generate_embeddings(texts)
    
    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.dimension,)


def test_generate_chunk_embedding():
    """Test embedding generation for code chunk."""
    embedder = EmbeddingGenerator()
    
    chunk = CodeChunk(
        content="def hello():\n    print('hello')",
        file_path="test.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=33,
        chunk_type="function_definition",
        language="python",
        metadata={"name": "hello"},
    )
    
    embedding = embedder.generate_chunk_embedding(chunk)
    
    assert embedding is not None
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedder.dimension,)


def test_enhance_chunk_text():
    """Test chunk text enhancement."""
    embedder = EmbeddingGenerator()
    
    chunk = CodeChunk(
        content="def test():\n    pass",
        file_path="test.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=20,
        chunk_type="function_definition",
        language="python",
        metadata={"name": "test"},
    )
    
    enhanced = embedder._enhance_chunk_text(chunk)
    
    assert "Language: python" in enhanced
    assert "Type: function_definition" in enhanced
    assert "Name: test" in enhanced
    assert chunk.content in enhanced
