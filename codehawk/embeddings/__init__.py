"""Embedding generation for code chunks."""

import logging
from typing import List, Optional, Protocol, Union

import os
from typing import List, Optional
import numpy as np

from codehawk.chunker import CodeChunk
from codehawk.config import settings

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Protocol for embedding backends."""

    dimension: int

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True):
        """Encode text or list of texts into embeddings."""


class DeterministicEmbeddingBackend:
    """Lightweight deterministic backend for offline mode."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    def _encode_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(self.dimension).astype(np.float32)

    def encode(self, texts: Union[str, List[str]], convert_to_numpy: bool = True):
        if isinstance(texts, str):
            return self._encode_text(texts)
        return np.stack([self._encode_text(text) for text in texts])


class EmbeddingGenerator:
    """Generates embeddings for code chunks."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        offline_mode: Optional[bool] = None,
        backend: Optional[EmbeddingBackend] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use
            offline_mode: Force offline mode and use deterministic fallback backend
            backend: Optional custom embedding backend (ggml/onnx compatible)
        """
        self.model_name = model_name
        self.offline_mode = (
            settings.embedding_offline_mode if offline_mode is None else offline_mode
        )
        self.model: Optional[EmbeddingBackend] = None
        self._backend_override = backend
        self.dimension = settings.embedding_dimension

        if backend is not None and hasattr(backend, "dimension"):
            self.dimension = int(getattr(backend, "dimension"))

    def load_model(self):
        """Load the embedding model or fallback backend."""
        if self.model is not None:
            return

        if self._backend_override is not None:
            self.model = self._backend_override
            logger.info("Using custom embedding backend override")
            return

        if self._try_load_sentence_transformer():
            return

        if self.offline_mode:
            self._load_offline_fallback()

        if self.model is None:
            raise RuntimeError(
                "Embedding model could not be loaded. Enable offline mode or provide a backend."
            )

    def _try_load_sentence_transformer(self) -> bool:
        if os.environ.get("CODEHAWK_DISABLE_MODEL_LOAD") == "1":
            logger.info("Embedding model loading disabled; using mock embeddings")
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not available; cannot load default model")
            return False

        try:
            init_kwargs = {"local_files_only": self.offline_mode}
            self.model = SentenceTransformer(self.model_name, **init_kwargs)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading embedding model {self.model_name}: {e}")
            return False

    def _load_offline_fallback(self):
        self.model = DeterministicEmbeddingBackend(self.dimension)
        logger.info("Using deterministic offline embedding backend")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.model is None:
            self.load_model()

        if self.model is None:
            logger.warning("Embedding model not loaded, returning mock embedding")
            return self._mock_embedding(text)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return np.asarray(embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {e}") from e

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts, with deterministic fallbacks when needed.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.model is None:
            try:
                self.load_model()
            except RuntimeError as e:
                logger.warning(
                    "Embedding model could not be loaded (%s); returning mock embeddings", e
                )
                return [self._mock_embedding(text) for text in texts]

        if self.model is None:
            logger.warning("Embedding model not loaded, returning mock embeddings")
            return [self._mock_embedding(text) for text in texts]

        try:
            results: List[np.ndarray] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                batch_array = np.asarray(batch_embeddings, dtype=np.float32)
                results.extend(batch_array)

            return results
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [self._mock_embedding(text) for text in texts]

    def generate_chunk_embedding(self, chunk: CodeChunk) -> np.ndarray:
        """
        Generate embedding for a code chunk.

        Args:
            chunk: Code chunk to embed

        Returns:
            Embedding vector
        """
        enhanced_text = self._enhance_chunk_text(chunk)
        return self.generate_embedding(enhanced_text)

    def _enhance_chunk_text(self, chunk: CodeChunk) -> str:
        """
        Enhance chunk text with metadata for better embeddings.

        Args:
            chunk: Code chunk

        Returns:
            Enhanced text
        """
        parts = [
            f"Language: {chunk.language}",
            f"Type: {chunk.chunk_type}",
            chunk.content,
        ]

        if "name" in chunk.metadata:
            parts.insert(1, f"Name: {chunk.metadata['name']}")

        return "\n".join(parts)

    def _mock_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic mock embedding when the model is unavailable."""
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(self.dimension, dtype=np.float32)
