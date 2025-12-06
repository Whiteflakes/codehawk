"""Context engine that orchestrates all components."""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime

from codehawk.parser import TreeSitterParser
from codehawk.chunker import CodeChunker, CodeChunk
from codehawk.embeddings import EmbeddingGenerator
from codehawk.database import Database
from codehawk.graph import GraphAnalyzer
from codehawk.lineage import LineageTracker
from codehawk.config import settings

logger = logging.getLogger(__name__)


class ContextEngine:
    """Main engine for code context generation and search."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the context engine.

        Args:
            database_url: Optional database connection string
            embedding_model: Optional embedding model name
        """
        self.database_url = database_url or settings.database_url
        self.embedding_model = embedding_model or settings.embedding_model

        self.parser = TreeSitterParser()
        self.chunker = CodeChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.embedder = EmbeddingGenerator(model_name=self.embedding_model)
        self.graph_analyzer = GraphAnalyzer()
        
        self.db: Optional[Database] = None

    def initialize(self):
        """Initialize the context engine."""
        logger.info("Initializing context engine")
        
        # Initialize database
        self.db = Database(self.database_url)
        self.db.connect()
        self.db.initialize_schema()
        
        # Load embedding model
        self.embedder.load_model()
        
        logger.info("Context engine initialized")

    def shutdown(self):
        """Shutdown the context engine."""
        if self.db:
            self.db.disconnect()
        logger.info("Context engine shutdown")

    def index_repository(self, repo_path: Path, repo_url: Optional[str] = None) -> int:
        """
        Index a repository.

        Args:
            repo_path: Path to repository
            repo_url: Optional repository URL

        Returns:
            Repository ID
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        logger.info(f"Indexing repository: {repo_path}")

        # Create repository record
        repo_name = repo_path.name
        repo_url = repo_url or f"file://{repo_path}"
        repo_id = self.db.insert_repository(repo_url, repo_name, str(repo_path))

        # Index commit history
        try:
            lineage = LineageTracker(repo_path)
            lineage.open_repository()
            commits = lineage.get_recent_commits(limit=1000)
            
            for commit_data in commits:
                commit_id = self.db.insert_commit(
                    repository_id=repo_id,
                    commit_hash=commit_data["hash"],
                    author=commit_data["author"],
                    message=commit_data["message"],
                    timestamp=commit_data["timestamp"],
                )
        except Exception as e:
            logger.warning(f"Error indexing commit history: {e}")

        # Index files
        try:
            self.index_directory(repo_id, repo_path)
        except Exception as e:
            logger.error(f"Error indexing repository files: {e}")
        logger.info(f"Indexed repository contents for {repo_name}")
        return repo_id

    def index_directory(self, repository_id: int, repo_path: Path) -> None:
        """Index a directory with change detection and cleanup."""
        if not self.db:
            raise RuntimeError("Database not initialized")

        current_files = self._get_code_files(repo_path)
        current_paths = {str(path.relative_to(repo_path)) for path in current_files}

        existing_files = self.db.get_files_for_repository(repository_id)
        missing_paths = set(existing_files.keys()) - current_paths

        if missing_paths:
            logger.info(f"Removing {len(missing_paths)} deleted files from index")
            self.db.delete_files(repository_id, list(missing_paths))

        for file_path in current_files:
            relative_path = str(file_path.relative_to(repo_path))
            content_bytes = file_path.read_bytes()
            content_hash = self._hash_content(content_bytes)
            modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)

            previous = existing_files.get(relative_path)
            if previous and previous.get("content_hash") == content_hash:
                prev_mtime = previous.get("modified_at")
                if prev_mtime and prev_mtime.replace(microsecond=0) == modified_at.replace(microsecond=0):
                    logger.debug(f"Skipping unchanged file: {relative_path}")
                    continue

            try:
                source_code = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(f"Could not decode file, skipping: {relative_path}")
                continue

            self.index_file(
                repository_id,
                file_path,
                repo_path,
                source_code=source_code,
                content_hash=content_hash,
                modified_at=modified_at,
            )

    def index_file(
        self,
        repository_id: int,
        file_path: Path,
        repo_path: Path,
        source_code: Optional[str] = None,
        content_hash: Optional[str] = None,
        modified_at: Optional[datetime] = None,
    ):
        """
        Index a single file.

        Args:
            repository_id: Repository ID
            file_path: Path to file
            repo_path: Path to repository root
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        # Get relative path
        relative_path = file_path.relative_to(repo_path)
        
        # Detect language
        language = self.parser._detect_language(file_path)
        if language == "unknown" or language not in settings.supported_languages:
            return

        # Read file
        try:
            if source_code is None:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return

        content_hash = content_hash or self._hash_content(source_code.encode("utf-8"))
        modified_at = modified_at or datetime.fromtimestamp(file_path.stat().st_mtime)

        # Check file size
        if len(source_code) > settings.max_file_size:
            logger.warning(f"File too large, skipping: {file_path}")
            return

        # Create file record
        file_id = self.db.insert_file(
            repository_id=repository_id,
            path=str(relative_path),
            language=language,
            size_bytes=len(source_code.encode("utf-8")),
            content_hash=content_hash,
            modified_at=modified_at,
        )

        # Parse file
        tree = self.parser.parse_file(file_path, language)

        # Chunk file
        chunks = self.chunker.chunk_file(file_path, tree, language, source_code)

        # Generate embeddings and store chunks
        enhanced_chunks: List[Tuple[CodeChunk, str]] = [
            (chunk, self.embedder._enhance_chunk_text(chunk)) for chunk in chunks
        ]
        embeddings = self.embedder.generate_embeddings([text for _, text in enhanced_chunks])

        chunk_records = []
        filtered_chunks: List[CodeChunk] = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is None:
                continue
            filtered_chunks.append(chunk)
            chunk_records.append(
                {
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "start_byte": chunk.start_byte,
                    "end_byte": chunk.end_byte,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "metadata": chunk.metadata,
                    "embedding": embedding,
                }
            )

        chunk_ids = self.db.insert_chunks_batch(file_id=file_id, chunks=chunk_records)

        # Analyze relationships
        if chunk_ids:
            relations = self.graph_analyzer.analyze_imports(filtered_chunks, chunk_ids)
            relations.extend(self.graph_analyzer.analyze_calls(filtered_chunks, chunk_ids))
            relations.extend(self.graph_analyzer.analyze_inheritance(filtered_chunks, chunk_ids))

            for relation in relations:
                try:
                    self.db.insert_relation(
                        source_chunk_id=relation.source_chunk_id,
                        target_chunk_id=relation.target_chunk_id,
                        relation_type=relation.relation_type,
                        metadata=relation.metadata,
                    )
                except Exception as e:
                    logger.debug(f"Error inserting relation: {e}")

    def search(
        self,
        query: str,
        limit: int = 10,
        repository_id: Optional[int] = None,
        repository_ids: Optional[List[int]] = None,
        languages: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        use_lexical: bool = False,
        lexical_weight: float = 0.3,
        reranker: Optional[Callable[[List[Dict[str, Any]], str], List[float]]] = None,
        rerank_weight: float = 0.4,
        rerank_top_k: Optional[int] = None,
        deduplicate: bool = True,
        dedup_threshold: float = 0.97,
        boost_top_level: float = 0.05,
        boost_recent_edit: float = 0.05,
        filter_match_boost: float = 0.02,
    ) -> List[Dict[str, Any]]:
        """
        Search for code chunks similar to query.

        Args:
            query: Search query
            limit: Maximum number of results
            repository_id: Optional repository filter (deprecated, use repository_ids)
            repository_ids: Optional list of repository IDs
            languages: Optional list of languages to filter
            since: Optional recency cutoff timestamp
            use_lexical: Blend vector search with lexical/BM25 scoring
            lexical_weight: Weight to give lexical scoring when blending
            reranker: Optional cross-encoder reranker callable
            rerank_weight: Blend factor for reranker scores
            rerank_top_k: Number of candidates to send to reranker
            deduplicate: Drop near-identical snippets
            dedup_threshold: Similarity threshold for deduplication
            boost_top_level: Bonus for top-level definitions
            boost_recent_edit: Bonus for recent edits
            filter_match_boost: Bonus for repo/language filter matches

        Returns:
            List of matching chunks
        """
        if not self.db:
            raise RuntimeError("Database not initialized")

        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query)
        if query_embedding is None:
            return []

        # Search database
        results = self.db.search_chunks(
            query_embedding,
            limit=limit,
            repository_id=repository_id,
            repository_ids=repository_ids,
            languages=languages,
            since=since,
            use_lexical=use_lexical,
            lexical_weight=lexical_weight,
            query_text=query,
            reranker=reranker,
            rerank_weight=rerank_weight,
            rerank_top_k=rerank_top_k,
            deduplicate=deduplicate,
            dedup_threshold=dedup_threshold,
            boost_top_level=boost_top_level,
            boost_recent_edit=boost_recent_edit,
            filter_match_boost=filter_match_boost,
        )
        return results

    def get_context_pack(
        self,
        query: str,
        limit: int = 5,
        include_relations: bool = True,
        include_lineage: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a context pack for LLM.

        Args:
            query: Query for context
            limit: Number of chunks to include
            include_relations: Include related chunks
            include_lineage: Include commit lineage

        Returns:
            Context pack dictionary
        """
        results = self.search(query, limit=limit)

        context_pack = {
            "query": query,
            "chunks": results,
            "relations": [],
            "lineage": [],
        }

        chunk_ids = [chunk.get("id") for chunk in results if chunk.get("id")]

        if include_relations and self.db:
            try:
                context_pack["relations"] = self.db.get_relations_for_chunks(chunk_ids)
            except Exception as e:
                logger.debug(f"Error fetching relations for context pack: {e}")

        if include_lineage and self.db:
            try:
                context_pack["lineage"] = self.db.get_lineage_for_chunks(chunk_ids)
            except Exception as e:
                logger.debug(f"Error fetching lineage for context pack: {e}")

        return context_pack

    def _get_code_files(self, directory: Path) -> List[Path]:
        """
        Get all code files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of code file paths
        """
        code_files = []
        extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs"}

        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in [
                "node_modules", "venv", "env", "__pycache__", "build", "dist", "target"
            ]]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in extensions:
                    code_files.append(file_path)

        return code_files

    @staticmethod
    def _hash_content(content: bytes) -> str:
        """Generate a stable hash for file content."""
        return hashlib.sha256(content).hexdigest()
