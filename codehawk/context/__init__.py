"""Context engine that orchestrates all components."""

import hashlib
import logging
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple
import os
from datetime import datetime

from codehawk.parser import TreeSitterParser
from codehawk.chunker import CodeChunker, CodeChunk
from codehawk.embeddings import EmbeddingGenerator
from codehawk.database import Database
from codehawk.graph import CodeRelation, GraphAnalyzer, GraphUpdatePlanner
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
        self.graph_update_planner = GraphUpdatePlanner(self.graph_analyzer)
        
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
        fingerprint: Optional[str] = None,
        reconcile: bool = False,
        source_code: Optional[str] = None,
        content_hash: Optional[str] = None,
        modified_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
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

        fingerprint = fingerprint or self._compute_fingerprint(repo_path, file_path, source_code)

        file_id = None
        if reconcile and fingerprint:
            existing = self.db.find_file_by_fingerprint(repository_id, fingerprint)
            if existing:
                file_id = existing["id"]
                if existing["path"] != str(relative_path):
                    self.db.update_file_path(file_id, str(relative_path))

        # Create or update file record
        file_id = file_id or self.db.insert_file(
            repository_id=repository_id,
            path=str(relative_path),
            language=language,
            size_bytes=len(source_code.encode("utf-8")),
            fingerprint=fingerprint,
            content_hash=content_hash,
            modified_at=modified_at,
        )

        if reconcile:
            # Clean stale graph + lineage data before re-inserting
            existing_chunk_ids = self.db.get_chunk_ids_for_file(file_id)
            if existing_chunk_ids:
                self.db.delete_relations_for_chunks(existing_chunk_ids)
            self.db.delete_lineage_for_files([file_id])
            self.db.delete_chunks_for_file(file_id)

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
                    "skeleton_content": chunk.skeleton,
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

        # Capture global symbols/constants for ranking boosts
        try:
            globals_found = self._extract_global_symbols(source_code, language)
            for chunk, chunk_id in zip(filtered_chunks, chunk_ids):
                for symbol in globals_found:
                    self.db.insert_global_symbol(
                        file_id=file_id,
                        chunk_id=chunk_id,
                        name=symbol,
                        metadata={"file_path": str(relative_path)},
                    )
        except Exception as exc:  # pragma: no cover - defensive best-effort
            logger.debug(f"Skipping global symbol capture for {file_path}: {exc}")

        self._record_file_lineage(repo_path, file_path, repository_id, file_id)

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

        return {"file_id": file_id, "chunk_ids": chunk_ids}

    def update_changed_files(
        self,
        repository_id: int,
        repo_path: Path,
        base_commit: str,
        target_commit: str = "HEAD",
    ) -> Dict[str, Any]:
        """Re-index only files touched between two commits with reconciliation."""

        lineage = LineageTracker(repo_path)
        lineage.open_repository()
        changes = lineage.detect_changes(base_commit, target_commit)

        changed_chunk_ids: List[int] = []
        file_ids: List[int] = []

        for change in changes:
            file_path = repo_path / change.path
            result = self.index_file(
                repository_id,
                file_path,
                repo_path,
                fingerprint=change.fingerprint,
                reconcile=True,
            )
            changed_chunk_ids.extend(result.get("chunk_ids", []))
            if result.get("file_id"):
                file_ids.append(result["file_id"])

        # Identify ancestors to re-run graph analysis on
        existing_relations = self.db.get_relations_by_target_ids(changed_chunk_ids)
        relation_objects = [
            CodeRelation(
                source_chunk_id=rel["source_chunk_id"],
                target_chunk_id=rel["target_chunk_id"],
                relation_type=rel["relation_type"],
                metadata=rel.get("metadata", {}),
            )
            for rel in existing_relations
        ]

        impacted_chunk_ids = self.graph_update_planner.impacted_chunks(
            relation_objects, set(changed_chunk_ids)
        )
        if impacted_chunk_ids:
            self.db.delete_relations_for_chunks(list(impacted_chunk_ids))

        return {
            "changes": changes,
            "impacted_chunk_ids": impacted_chunk_ids,
            "file_ids": file_ids,
        }

    def _record_file_lineage(
        self, repo_path: Path, file_path: Path, repository_id: int, file_id: int
    ) -> None:
        """Link the current file contents to the latest commit when git metadata is available."""

        if not self.db or not hasattr(self.db, "insert_commit"):
            return

        try:
            commit_info = self._resolve_file_commit(repo_path, file_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(f"Unable to resolve commit for {file_path}: {exc}")
            return

        if not commit_info:
            return

        try:
            commit_id = self.db.insert_commit(
                repository_id=repository_id,
                commit_hash=commit_info["hash"],
                author=commit_info.get("author", ""),
                message=commit_info.get("message", ""),
                timestamp=commit_info.get("timestamp"),
            )
            self.db.link_file_commit(
                file_id=file_id,
                commit_id=commit_id,
                change_type=commit_info.get("change_type", "modified"),
            )
        except Exception as exc:  # pragma: no cover - best-effort lineage capture
            logger.debug(f"Unable to link commit for {file_path}: {exc}")

    def _resolve_file_commit(
        self, repo_path: Path, file_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Return latest commit info for a file, safely handling non-git repos."""

        git_dir = repo_path / ".git"
        if not git_dir.exists() or not git_dir.is_dir():
            return None

        try:
            lineage = LineageTracker(repo_path)
            lineage.open_repository()
            commits = lineage.get_file_commits(file_path, limit=1)
            return commits[0] if commits else None
        except Exception as exc:  # pragma: no cover - depends on git availability
            logger.debug(f"Skipping lineage linking for {file_path}: {exc}")
            return None

    def _extract_global_symbols(self, source_code: str, language: str) -> List[str]:
        """Identify module-level constants/configs for boosting."""

        symbols: List[str] = []
        if language == "python":
            import ast

            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                return symbols

            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            symbols.append(target.id)
                elif isinstance(node, ast.AnnAssign):
                    target = node.target
                    if isinstance(target, ast.Name) and target.id.isupper():
                        symbols.append(target.id)

        return symbols

    def _compute_fingerprint(
        self, repo_path: Path, file_path: Path, content: str
    ) -> Optional[str]:
        try:
            lineage = LineageTracker(repo_path)
            lineage.open_repository()
            return lineage.get_blob_fingerprint(file_path)
        except Exception:
            import hashlib

            return hashlib.sha1(content.encode("utf-8")).hexdigest()

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
        graph_hops: int = 1,
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

        # Boost global symbols/config constants when query tokens reference them
        query_terms = [token.strip(" ,.:()") for token in query.split() if token]
        if query_terms:
            try:
                global_hits = self.db.search_global_symbols(query_terms, repository_ids or ([repository_id] if repository_id else None))
                for hit in global_hits:
                    results.insert(
                        0,
                        {
                            "id": hit.get("chunk_id"),
                            "content": hit.get("content"),
                            "skeleton_content": hit.get("skeleton_content"),
                            "file_path": hit.get("file_path"),
                            "metadata": hit.get("metadata", {}),
                            "chunk_type": "global",
                            "language": hit.get("language"),
                            "boost_reason": "global_symbol",
                            "combined_score": 1.5,
                        },
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Global symbol search failed: {exc}")
        if graph_hops > 0:
            try:
                expansions = self._expand_with_graph(results, graph_hops)
                results.extend(expansions)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Graph expansion failed: {exc}")

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

    def _expand_with_graph(self, seeds: List[Dict[str, Any]], hops: int) -> List[Dict[str, Any]]:
        """Return related chunks using graph edges (skeleton view for neighbors)."""

        if not self.db or not seeds:
            return []

        hop_ids = {chunk.get("id") for chunk in seeds if chunk.get("id")}
        all_related: List[Dict[str, Any]] = []

        for _ in range(hops):
            relations = self.db.get_relations_for_chunks(list(hop_ids))
            neighbor_ids = set()
            for rel in relations:
                neighbor_ids.add(rel["source_chunk_id"])
                neighbor_ids.add(rel["target_chunk_id"])
            neighbor_ids -= hop_ids
            if not neighbor_ids:
                break

            neighbor_chunks = self.db.get_chunks_by_ids(list(neighbor_ids))
            for chunk in neighbor_chunks:
                original_content = chunk.get("content")
                chunk["full_content"] = original_content
                chunk["content"] = chunk.get("skeleton_content") or original_content
                chunk["relation_type"] = "graph_neighbor"
                chunk["skeleton_only"] = True
                all_related.append(chunk)
            hop_ids.update(neighbor_ids)

        return all_related

    def get_repository_overview(self, repository_id: int) -> List[Dict[str, Any]]:
        """Expose a skeletonized repository overview for passive MCP resources."""

        if not self.db:
            raise RuntimeError("Database not initialized")
        return self.db.get_repository_skeletons(repository_id)

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
