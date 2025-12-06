"""Tree-sitter parser for code analysis."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import tree_sitter

logger = logging.getLogger(__name__)


class TreeSitterParser:
    """Parser using tree-sitter for multi-language code analysis."""

    def __init__(self):
        """Initialize the tree-sitter parser."""
        self.parsers: Dict[str, tree_sitter.Parser] = {}
        self.languages: Dict[str, tree_sitter.Language] = {}
        self._initialize_languages()

    def _initialize_languages(self):
        """Initialize tree-sitter languages."""
        # Note: In production, you would build language binaries
        # For now, we'll handle gracefully if languages aren't available
        try:
            import tree_sitter_python as tspython
            self.languages["python"] = tree_sitter.Language(tspython.language())
            parser = tree_sitter.Parser(self.languages["python"])
            self.parsers["python"] = parser
        except ImportError:
            logger.warning("tree-sitter-python not available")

        try:
            import tree_sitter_javascript as tsjavascript
            self.languages["javascript"] = tree_sitter.Language(tsjavascript.language())
            parser = tree_sitter.Parser(self.languages["javascript"])
            self.parsers["javascript"] = parser
        except ImportError:
            logger.warning("tree-sitter-javascript not available")

        try:
            import tree_sitter_typescript as tstypescript
            self.languages["typescript"] = tree_sitter.Language(tstypescript.language())
            parser = tree_sitter.Parser(self.languages["typescript"])
            self.parsers["typescript"] = parser
        except ImportError:
            logger.warning("tree-sitter-typescript not available")

    def parse_file(self, file_path: Path, language: Optional[str] = None) -> Optional[tree_sitter.Tree]:
        """
        Parse a file using tree-sitter.

        Args:
            file_path: Path to the file to parse
            language: Optional language override

        Returns:
            Parsed tree or None if parsing fails
        """
        if language is None:
            language = self._detect_language(file_path)

        if language not in self.parsers:
            logger.warning(f"Language {language} not supported")
            return None

        try:
            with open(file_path, "rb") as f:
                source_code = f.read()
            return self.parsers[language].parse(source_code)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def parse_code(self, code: str, language: str) -> Optional[tree_sitter.Tree]:
        """
        Parse code string using tree-sitter.

        Args:
            code: Source code string
            language: Programming language

        Returns:
            Parsed tree or None if parsing fails
        """
        if language not in self.parsers:
            logger.warning(f"Language {language} not supported")
            return None

        try:
            return self.parsers[language].parse(bytes(code, "utf-8"))
        except Exception as e:
            logger.error(f"Error parsing code: {e}")
            return None

    def _detect_language(self, file_path: Path) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected language
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        return extension_map.get(file_path.suffix, "unknown")

    def extract_functions(self, tree: tree_sitter.Tree, language: str) -> List[Dict[str, Any]]:
        """
        Extract function definitions from parsed tree.

        Args:
            tree: Parsed tree-sitter tree
            language: Programming language

        Returns:
            List of function information dictionaries
        """
        functions = []
        
        if language == "python":
            query_str = """
            (function_definition
                name: (identifier) @name
                parameters: (parameters) @params
                body: (block) @body) @function
            """
        elif language in ["javascript", "typescript"]:
            query_str = """
            (function_declaration
                name: (identifier) @name
                parameters: (formal_parameters) @params
                body: (statement_block) @body) @function
            """
        else:
            return functions

        try:
            if language in self.languages:
                query = self.languages[language].query(query_str)
                captures = query.captures(tree.root_node)
                
                for node, capture_name in captures:
                    if capture_name == "function":
                        functions.append({
                            "name": node.child_by_field_name("name").text.decode("utf-8") if node.child_by_field_name("name") else "anonymous",
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "start_point": node.start_point,
                            "end_point": node.end_point,
                        })
        except Exception as e:
            logger.error(f"Error extracting functions: {e}")

        return functions

    def extract_classes(self, tree: tree_sitter.Tree, language: str) -> List[Dict[str, Any]]:
        """
        Extract class definitions from parsed tree.

        Args:
            tree: Parsed tree-sitter tree
            language: Programming language

        Returns:
            List of class information dictionaries
        """
        classes = []

        if language == "python":
            query_str = """
            (class_definition
                name: (identifier) @name
                body: (block) @body) @class
            """
        elif language in ["javascript", "typescript"]:
            query_str = """
            (class_declaration
                name: (identifier) @name
                body: (class_body) @body) @class
            """
        else:
            return classes

        try:
            if language in self.languages:
                query = self.languages[language].query(query_str)
                captures = query.captures(tree.root_node)
                
                for node, capture_name in captures:
                    if capture_name == "class":
                        classes.append({
                            "name": node.child_by_field_name("name").text.decode("utf-8") if node.child_by_field_name("name") else "anonymous",
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "start_point": node.start_point,
                            "end_point": node.end_point,
                        })
        except Exception as e:
            logger.error(f"Error extracting classes: {e}")

        return classes
