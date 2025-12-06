"""Unit tests for embedding generator."""

import pytest
import numpy as np

from codehawk.embeddings import EmbeddingGenerator
from codehawk.chunker import CodeChunk


@pytest.fixture
def offline_embedder():
    """Create an embedder forced into offline mode."""

    embedder = EmbeddingGenerator(offline_mode=True)
    # Avoid actually loading sentence-transformers when offline testing
    embedder._try_load_sentence_transformer = lambda: False  # type: ignore[attr-defined]
    return embedder


def test_embedder_initialization(offline_embedder):
    """Test embedder initialization."""
    assert offline_embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert offline_embedder.dimension == 384


def test_generate_embedding(offline_embedder):
    """Test embedding generation."""

    text = "This is a test string for embedding"
    embedding = offline_embedder.generate_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (offline_embedder.dimension,)
    assert embedding.dtype == np.float32


def test_generate_embeddings_batch(offline_embedder):
    """Test batch embedding generation."""

    texts = [
        "First test string",
        "Second test string",
        "Third test string",
    ]

    embeddings = offline_embedder.generate_embeddings(texts)

    assert len(embeddings) == len(texts)
    for embedding in embeddings:
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (offline_embedder.dimension,)


def test_generate_chunk_embedding(offline_embedder):
    """Test embedding generation for code chunk."""

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

    embedding = offline_embedder.generate_chunk_embedding(chunk)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (offline_embedder.dimension,)


def test_enhance_chunk_text():
    """Test chunk text enhancement."""
    embedder = EmbeddingGenerator(offline_mode=True)
    
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


def test_offline_fallback_when_download_fails(monkeypatch):
    """Offline mode should fallback to deterministic backend when download fails."""

    embedder = EmbeddingGenerator(offline_mode=True)
    monkeypatch.setattr(embedder, "_try_load_sentence_transformer", lambda: False)

    embeddings = embedder.generate_embeddings(["a", "b"])

    assert len(embeddings) == 2
    assert all(isinstance(embedding, np.ndarray) for embedding in embeddings)


def test_error_surface_when_no_backend_available(monkeypatch):
    """Without offline mode, failures should raise instead of returning None."""

    embedder = EmbeddingGenerator(offline_mode=False)
    monkeypatch.setattr(embedder, "_try_load_sentence_transformer", lambda: False)

    with pytest.raises(RuntimeError):
        embedder.generate_embedding("text")
