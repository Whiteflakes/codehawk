"""Embedding generation for code chunks."""

import logging
from typing import List, Optional
import numpy as np
from codehawk.chunker import CodeChunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for code chunks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2

    def load_model(self):
        """Load the embedding model."""
        if self.model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if generation fails
        """
        if self.model is None:
            self.load_model()

        if self.model is None:
            # Return mock embedding for testing
            return np.random.rand(self.dimension).astype(np.float32)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load_model()

        if self.model is None:
            # Return mock embeddings for testing
            return [np.random.rand(self.dimension).astype(np.float32) for _ in texts]

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [None] * len(texts)

    def generate_chunk_embedding(self, chunk: CodeChunk) -> Optional[np.ndarray]:
        """
        Generate embedding for a code chunk.

        Args:
            chunk: Code chunk to embed

        Returns:
            Embedding vector or None if generation fails
        """
        # Enhance chunk content with metadata
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

        # Add metadata if available
        if "name" in chunk.metadata:
            parts.insert(1, f"Name: {chunk.metadata['name']}")

        return "\n".join(parts)
