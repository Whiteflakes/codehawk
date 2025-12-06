"""CodeHawk - Open-source code context engine."""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "ContextEngine":
        from codehawk.context.engine import ContextEngine
        return ContextEngine
    elif name == "TreeSitterParser":
        from codehawk.parser.tree_sitter_parser import TreeSitterParser
        return TreeSitterParser
    elif name == "CodeChunker":
        from codehawk.chunker.code_chunker import CodeChunker
        return CodeChunker
    elif name == "EmbeddingGenerator":
        from codehawk.embeddings.generator import EmbeddingGenerator
        return EmbeddingGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ContextEngine",
    "TreeSitterParser",
    "CodeChunker",
    "EmbeddingGenerator",
]
