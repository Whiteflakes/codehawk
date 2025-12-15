"""Graph-based code relations analysis."""

import ast
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from codehawk.chunker import CodeChunk
from codehawk.parser import TreeSitterParser

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
        self.parser = TreeSitterParser()
        self._symbol_index_cache: Optional[Tuple[Tuple[int, ...], Dict[str, Dict[str, Set[int]]]]] = None

    def analyze_imports(self, chunks: List[CodeChunk], chunk_ids: List[int]) -> List[CodeRelation]:
        """
        Analyze import relationships between chunks.

        Args:
            chunks: List of code chunks
            chunk_ids: List of corresponding chunk IDs in database

        Returns:
            List of relations
        """
        relations: List[CodeRelation] = []
        symbol_index = self._get_symbol_index(chunks, chunk_ids)
        seen: Set[Tuple[int, int, str]] = set()

        for i, chunk in enumerate(chunks):
            imports = self._extract_imports(chunk.content, chunk.language)

            for imp in imports:
                target_ids: Set[int] = set()
                module = imp.get("module")
                name = imp.get("name") or imp.get("import_name")

                if module and module in symbol_index["modules"]:
                    target_ids.update(symbol_index["modules"].get(module, set()))

                if name:
                    target_ids.update(symbol_index["functions"].get(name, set()))
                    target_ids.update(symbol_index["classes"].get(name, set()))

                for target_id in target_ids:
                    key = (chunk_ids[i], target_id, "imports")
                    if target_id != chunk_ids[i] and key not in seen:
                        seen.add(key)
                        relations.append(
                            CodeRelation(
                                source_chunk_id=chunk_ids[i],
                                target_chunk_id=target_id,
                                relation_type="imports",
                                metadata={
                                    "import_name": name or module,
                                    "source_file": chunk.file_path,
                                    "target_file": self._symbol_index_file(symbol_index, target_id),
                                },
                            )
                        )

        return relations

    def analyze_calls(self, chunks: List[CodeChunk], chunk_ids: List[int]) -> List[CodeRelation]:
        """
        Analyze function call relationships.

        Args:
            chunks: List of code chunks
            chunk_ids: List of corresponding chunk IDs in database

        Returns:
            List of relations
        """
        relations: List[CodeRelation] = []
        symbol_index = self._get_symbol_index(chunks, chunk_ids)
        seen: Set[Tuple[int, int, str]] = set()

        for i, chunk in enumerate(chunks):
            if chunk.chunk_type in ["function_definition", "method_definition", "decorated_definition"]:
                calls = self._extract_function_calls(chunk.content, chunk.language)

                for call in calls:
                    target_ids = symbol_index["functions"].get(call, set())
                    for target_id in target_ids:
                        key = (chunk_ids[i], target_id, "calls")
                        if target_id != chunk_ids[i] and key not in seen:
                            seen.add(key)
                            relations.append(
                                CodeRelation(
                                    source_chunk_id=chunk_ids[i],
                                    target_chunk_id=target_id,
                                    relation_type="calls",
                                    metadata={
                                        "function_name": call,
                                        "source_file": chunk.file_path,
                                        "target_file": self._symbol_index_file(symbol_index, target_id),
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
        relations: List[CodeRelation] = []
        symbol_index = self._get_symbol_index(chunks, chunk_ids)
        seen: Set[Tuple[int, int, str]] = set()

        for i, chunk in enumerate(chunks):
            if chunk.chunk_type == "class_definition":
                base_classes = self._extract_base_classes(chunk.content, chunk.language)

                for base_class in base_classes:
                    target_ids = symbol_index["classes"].get(base_class, set())
                    for target_id in target_ids:
                        key = (chunk_ids[i], target_id, "inherits")
                        if target_id != chunk_ids[i] and key not in seen:
                            seen.add(key)
                            relations.append(
                                CodeRelation(
                                    source_chunk_id=chunk_ids[i],
                                    target_chunk_id=target_id,
                                    relation_type="inherits",
                                    metadata={
                                        "class_name": base_class,
                                        "source_file": chunk.file_path,
                                        "target_file": self._symbol_index_file(symbol_index, target_id),
                                    },
                                )
                            )

        return relations

    def _get_symbol_index(
        self, chunks: List[CodeChunk], chunk_ids: List[int]
    ) -> Dict[str, Dict[str, Set[int]]]:
        """Build (or reuse) a repository-wide symbol index for definition resolution."""

        cache_key = (tuple(chunk_ids), tuple(hash(chunk.content) for chunk in chunks))
        if self._symbol_index_cache and self._symbol_index_cache[0] == cache_key:
            return self._symbol_index_cache[1]

        index: Dict[str, Dict[str, Set[int]]] = {
            "modules": {},
            "functions": {},
            "classes": {},
            "files": {},
        }

        for chunk, chunk_id in zip(chunks, chunk_ids):
            module_name = Path(chunk.file_path).stem
            index["modules"].setdefault(module_name, set()).add(chunk_id)
            index["files"][chunk_id] = chunk.file_path

            for func in self._extract_function_definitions(chunk):
                index["functions"].setdefault(func, set()).add(chunk_id)
            for cls in self._extract_class_definitions(chunk):
                index["classes"].setdefault(cls, set()).add(chunk_id)

        self._symbol_index_cache = (cache_key, index)
        return index

    def _symbol_index_file(self, symbol_index: Dict[str, Dict[str, Set[int]]], chunk_id: int) -> str:
        """Lookup file path for a chunk id from the symbol index."""

        return symbol_index.get("files", {}).get(chunk_id, "")

    def _extract_imports(self, content: str, language: str) -> List[Dict[str, str]]:
        """Extract import statements using syntax-aware parsing when possible."""

        if language == "python":
            try:
                tree = ast.parse(content)
                imports: List[Dict[str, str]] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({"module": alias.name.split(".")[0], "name": alias.asname or alias.name})
                    elif isinstance(node, ast.ImportFrom):
                        module = (node.module or "").split(".")[0]
                        for alias in node.names:
                            imports.append({"module": module, "name": alias.name})
                return imports
            except SyntaxError:
                logger.debug("Failed to parse imports with ast; falling back to heuristics")

        imports: List[Dict[str, str]] = []
        if language in ["javascript", "typescript"]:
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("import ") and "from" in line:
                    try:
                        module = line.split("from")[1].strip().strip("\"';")
                        name = (
                            line.split("import")[1]
                            .split("from")[0]
                            .strip()
                            .strip("{}")
                            .split(" as ")[0]
                        )
                        imports.append({"module": module.split("/")[-1], "name": name})
                    except Exception:
                        continue
        return imports

    def _extract_function_calls(self, content: str, language: str) -> List[str]:
        """Extract function calls using AST when available."""

        calls: Set[str] = set()

        if language == "python":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name):
                            calls.add(func.id)
                        elif isinstance(func, ast.Attribute):
                            calls.add(func.attr)
                return list(calls)
            except SyntaxError:
                logger.debug("Failed to parse function calls with ast; falling back to regex")

        import re

        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        matches = re.finditer(pattern, content)

        for match in matches:
            func_name = match.group(1)
            if func_name not in ["if", "while", "for", "def", "class", "function"]:
                calls.add(func_name)

        return list(calls)

    def _extract_base_classes(self, content: str, language: str) -> List[str]:
        """Extract base classes from class definition using AST when available."""

        if language == "python":
            try:
                tree = ast.parse(content)
                bases: List[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                bases.append(base.attr)
                return bases
            except SyntaxError:
                logger.debug("Failed to parse inheritance with ast; falling back to regex")

        import re

        pattern = r'class\s+\w+\s*\(([^)]+)\)'
        match = re.search(pattern, content)
        if not match:
            return []

        base_classes: List[str] = []
        bases = match.group(1).split(",")
        for base in bases:
            name = base.strip().split(".")[-1]
            if name and name != "object":
                base_classes.append(name)

        return base_classes

    def _extract_function_definitions(self, chunk: CodeChunk) -> Set[str]:
        """Extract defined function names from a chunk using syntax-aware parsing."""

        names: Set[str] = set()
        if chunk.language == "python":
            try:
                tree = ast.parse(chunk.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        names.add(node.name)
            except SyntaxError:
                logger.debug("Failed to parse function definitions with ast")

        tree = self.parser.parse_code(chunk.content, chunk.language) if hasattr(self.parser, "parse_code") else None
        if tree:
            try:
                for func in self.parser.extract_functions(tree, chunk.language):
                    names.add(func.get("name", ""))
            except Exception:
                logger.debug("Failed to extract functions via tree-sitter")

        if not names and chunk.chunk_type in ["function_definition", "method_definition", "decorated_definition"]:
            header = chunk.content.split("\n")[0]
            if "def " in header:
                candidate = header.split("def ")[1].split("(")[0].strip()
                if candidate:
                    names.add(candidate)

        return {name for name in names if name}

    def _extract_class_definitions(self, chunk: CodeChunk) -> Set[str]:
        """Extract defined class names from a chunk using syntax-aware parsing."""

        names: Set[str] = set()
        if chunk.language == "python":
            try:
                tree = ast.parse(chunk.content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        names.add(node.name)
            except SyntaxError:
                logger.debug("Failed to parse class definitions with ast")

        tree = self.parser.parse_code(chunk.content, chunk.language) if hasattr(self.parser, "parse_code") else None
        if tree:
            try:
                for cls in self.parser.extract_classes(tree, chunk.language):
                    names.add(cls.get("name", ""))
            except Exception:
                logger.debug("Failed to extract classes via tree-sitter")

        if not names and chunk.chunk_type == "class_definition":
            header = chunk.content.split("\n")[0]
            if "class " in header:
                candidate = header.split("class ")[1].split("(")[0].strip().strip(":")
                if candidate:
                    names.add(candidate)

        return {name for name in names if name}


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
