"""CodeHawk - Open-source code context engine."""

__version__ = "0.1.0"

from codehawk.context.engine import ContextEngine
from codehawk.parser.tree_sitter_parser import TreeSitterParser
from codehawk.chunker.code_chunker import CodeChunker
from codehawk.embeddings.generator import EmbeddingGenerator

__all__ = [
    "ContextEngine",
    "TreeSitterParser",
    "CodeChunker",
    "EmbeddingGenerator",
]
