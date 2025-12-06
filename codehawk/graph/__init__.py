"""Graph-based code relations analysis."""

import logging
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from codehawk.chunker import CodeChunk

logger = logging.getLogger(__name__)


@dataclass
class CodeRelation:
    """Represents a relation between code chunks."""

    source_chunk_id: int
    target_chunk_id: int
    relation_type: str
    metadata: Dict[str, Any]


class GraphAnalyzer:
    """Analyzes code relationships and builds dependency graphs."""

    def __init__(self):
        """Initialize the graph analyzer."""
        self.relations: List[CodeRelation] = []

    def analyze_imports(self, chunks: List[CodeChunk], chunk_ids: List[int]) -> List[CodeRelation]:
        """
        Analyze import relationships between chunks.

        Args:
            chunks: List of code chunks
            chunk_ids: List of corresponding chunk IDs in database

        Returns:
            List of relations
        """
        relations = []
        
        for i, chunk in enumerate(chunks):
            imports = self._extract_imports(chunk.content, chunk.language)
            
            for imp in imports:
                # Find chunks that might define this import
                for j, target_chunk in enumerate(chunks):
                    if i != j and self._chunk_defines(target_chunk, imp):
                        relations.append(
                            CodeRelation(
                                source_chunk_id=chunk_ids[i],
                                target_chunk_id=chunk_ids[j],
                                relation_type="imports",
                                metadata={
                                    "import_name": imp,
                                    "source_file": chunk.file_path,
                                    "target_file": target_chunk.file_path,
                                },
                            )
                        )

        return relations


class GraphUpdatePlanner:
    """Determines which chunks need to be re-analyzed after file changes."""

    def __init__(self, analyzer: "GraphAnalyzer"):
        self.analyzer = analyzer

    def impacted_chunks(
        self, relations: List[CodeRelation], changed_chunk_ids: Set[int]
    ) -> Set[int]:
        """Return changed chunks plus their ancestors in the relation graph."""

        impacted: Set[int] = set(changed_chunk_ids)
        added = True

        while added:
            added = False
            for relation in relations:
                if relation.target_chunk_id in impacted and relation.source_chunk_id not in impacted:
                    impacted.add(relation.source_chunk_id)
                    added = True

        return impacted

    def analyze_calls(self, chunks: List[CodeChunk], chunk_ids: List[int]) -> List[CodeRelation]:
        """
        Analyze function call relationships.

        Args:
            chunks: List of code chunks
            chunk_ids: List of corresponding chunk IDs in database

        Returns:
            List of relations
        """
        relations = []

        for i, chunk in enumerate(chunks):
            if chunk.chunk_type in ["function_definition", "method_definition"]:
                calls = self._extract_function_calls(chunk.content, chunk.language)
                
                for call in calls:
                    # Find chunks that define this function
                    for j, target_chunk in enumerate(chunks):
                        if i != j and self._is_function_definition(target_chunk, call):
                            relations.append(
                                CodeRelation(
                                    source_chunk_id=chunk_ids[i],
                                    target_chunk_id=chunk_ids[j],
                                    relation_type="calls",
                                    metadata={
                                        "function_name": call,
                                        "source_file": chunk.file_path,
                                        "target_file": target_chunk.file_path,
                                    },
                                )
                            )

        return relations

    def analyze_inheritance(self, chunks: List[CodeChunk], chunk_ids: List[int]) -> List[CodeRelation]:
        """
        Analyze class inheritance relationships.

        Args:
            chunks: List of code chunks
            chunk_ids: List of corresponding chunk IDs in database

        Returns:
            List of relations
        """
        relations = []

        for i, chunk in enumerate(chunks):
            if chunk.chunk_type == "class_definition":
                base_classes = self._extract_base_classes(chunk.content, chunk.language)
                
                for base_class in base_classes:
                    # Find chunks that define this base class
                    for j, target_chunk in enumerate(chunks):
                        if i != j and self._is_class_definition(target_chunk, base_class):
                            relations.append(
                                CodeRelation(
                                    source_chunk_id=chunk_ids[i],
                                    target_chunk_id=chunk_ids[j],
                                    relation_type="inherits",
                                    metadata={
                                        "class_name": base_class,
                                        "source_file": chunk.file_path,
                                        "target_file": target_chunk.file_path,
                                    },
                                )
                            )

        return relations

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        
        if language == "python":
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("import "):
                    parts = line.split()
                    if len(parts) >= 2:
                        imports.append(parts[1].split(".")[0])
                elif line.startswith("from "):
                    parts = line.split()
                    if len(parts) >= 2:
                        imports.append(parts[1].split(".")[0])
        
        elif language in ["javascript", "typescript"]:
            for line in content.split("\n"):
                line = line.strip()
                if "import " in line or "require(" in line:
                    # Simple extraction - could be improved with proper parsing
                    if "from" in line:
                        parts = line.split("from")
                        if len(parts) >= 2:
                            module = parts[1].strip().strip("\"';").split("/")[-1]
                            imports.append(module)

        return imports

    def _extract_function_calls(self, content: str, language: str) -> List[str]:
        """Extract function calls from code."""
        calls = []
        
        # Simple heuristic - look for patterns like "function_name("
        import re
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.finditer(pattern, content)
        
        for match in matches:
            func_name = match.group(1)
            # Filter out common keywords
            if func_name not in ["if", "while", "for", "def", "class", "function"]:
                calls.append(func_name)

        return list(set(calls))

    def _extract_base_classes(self, content: str, language: str) -> List[str]:
        """Extract base classes from class definition."""
        base_classes = []
        
        if language == "python":
            import re
            pattern = r'class\s+\w+\s*\(([^)]+)\)'
            match = re.search(pattern, content)
            if match:
                bases = match.group(1).split(",")
                for base in bases:
                    base = base.strip().split(".")[-1]
                    if base and base != "object":
                        base_classes.append(base)

        return base_classes

    def _chunk_defines(self, chunk: CodeChunk, name: str) -> bool:
        """Check if chunk defines a given name."""
        # Simple heuristic - check if name appears in chunk as a definition
        if chunk.chunk_type in ["function_definition", "class_definition"]:
            return name in chunk.content.split("\n")[0]
        return False

    def _is_function_definition(self, chunk: CodeChunk, name: str) -> bool:
        """Check if chunk is a function definition with given name."""
        if chunk.chunk_type in ["function_definition", "method_definition"]:
            first_line = chunk.content.split("\n")[0]
            return name in first_line
        return False

    def _is_class_definition(self, chunk: CodeChunk, name: str) -> bool:
        """Check if chunk is a class definition with given name."""
        if chunk.chunk_type == "class_definition":
            first_line = chunk.content.split("\n")[0]
            return name in first_line
        return False
