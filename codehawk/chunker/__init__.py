"""Code chunking for embedding generation."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import tree_sitter

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    chunk_type: str  # 'function', 'class', 'block', 'file'
    language: str
    metadata: Dict[str, Any]


class CodeChunker:
    """Chunks code into embeddable units."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize the code chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_file(
        self,
        file_path: Path,
        tree: Optional[tree_sitter.Tree],
        language: str,
        source_code: str,
    ) -> List[CodeChunk]:
        """
        Chunk a file into embeddable units.

        Args:
            file_path: Path to the file
            tree: Parsed tree-sitter tree (optional)
            language: Programming language
            source_code: Source code content

        Returns:
            List of code chunks
        """
        chunks = []

        if tree is not None:
            # Try to chunk by semantic units (functions, classes, etc.)
            chunks = self._chunk_by_structure(file_path, tree, language, source_code)

        if not chunks:
            # Fall back to simple line-based chunking
            chunks = self._chunk_by_lines(file_path, language, source_code)

        return chunks

    def _chunk_by_structure(
        self,
        file_path: Path,
        tree: tree_sitter.Tree,
        language: str,
        source_code: str,
    ) -> List[CodeChunk]:
        """
        Chunk code by semantic structure (functions, classes, etc.).

        Args:
            file_path: Path to the file
            tree: Parsed tree-sitter tree
            language: Programming language
            source_code: Source code content

        Returns:
            List of code chunks
        """
        chunks = []
        source_bytes = source_code.encode("utf-8")

        def traverse_node(node: tree_sitter.Node, depth: int = 0):
            """Recursively traverse tree nodes."""
            # Check if node is a chunkable unit
            if self._is_chunkable_node(node, language):
                start_byte = node.start_byte
                end_byte = node.end_byte
                content = source_bytes[start_byte:end_byte].decode("utf-8")

                # Only chunk if size is reasonable
                if len(content) <= self.chunk_size * 2:
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1

                    chunk = CodeChunk(
                        content=content,
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=end_line,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        chunk_type=node.type,
                        language=language,
                        metadata={
                            "node_type": node.type,
                            "depth": depth,
                        },
                    )
                    chunks.append(chunk)
                    return  # Don't traverse children if we chunked this node

            # Traverse children
            for child in node.children:
                traverse_node(child, depth + 1)

        traverse_node(tree.root_node)
        return chunks

    def _is_chunkable_node(self, node: tree_sitter.Node, language: str) -> bool:
        """
        Check if a node should be chunked.

        Args:
            node: Tree-sitter node
            language: Programming language

        Returns:
            True if node should be chunked
        """
        chunkable_types = {
            "python": [
                "function_definition",
                "class_definition",
                "decorated_definition",
            ],
            "javascript": [
                "function_declaration",
                "class_declaration",
                "method_definition",
                "arrow_function",
            ],
            "typescript": [
                "function_declaration",
                "class_declaration",
                "method_definition",
                "arrow_function",
            ],
        }

        return node.type in chunkable_types.get(language, [])

    def _chunk_by_lines(
        self,
        file_path: Path,
        language: str,
        source_code: str,
    ) -> List[CodeChunk]:
        """
        Chunk code by lines with overlap.

        Args:
            file_path: Path to the file
            language: Programming language
            source_code: Source code content

        Returns:
            List of code chunks
        """
        chunks = []
        lines = source_code.split("\n")
        
        # Calculate chunk size in lines (approximate)
        lines_per_chunk = max(1, self.chunk_size // 60)  # Assume ~60 chars per line
        overlap_lines = max(1, self.overlap // 60)

        start_idx = 0
        chunk_id = 0

        while start_idx < len(lines):
            end_idx = min(start_idx + lines_per_chunk, len(lines))
            chunk_lines = lines[start_idx:end_idx]
            content = "\n".join(chunk_lines)

            # Calculate byte positions
            preceding_content = "\n".join(lines[:start_idx])
            start_byte = len(preceding_content.encode("utf-8")) + (1 if start_idx > 0 else 0)
            end_byte = start_byte + len(content.encode("utf-8"))

            chunk = CodeChunk(
                content=content,
                file_path=str(file_path),
                start_line=start_idx + 1,
                end_line=end_idx,
                start_byte=start_byte,
                end_byte=end_byte,
                chunk_type="block",
                language=language,
                metadata={
                    "chunk_id": chunk_id,
                    "total_lines": len(lines),
                },
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            start_idx = end_idx - overlap_lines
            if start_idx >= len(lines) - overlap_lines:
                break
            chunk_id += 1

        return chunks
