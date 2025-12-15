"""Database schema and connection management."""

import logging
from difflib import SequenceMatcher
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import numpy as np

from codehawk.config import settings

logger = logging.getLogger(__name__)

# Optional dependency - fail gracefully if not available
try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg2 not available - database features will be disabled")


class Database:
    """PostgreSQL database with pgvector support."""

    def __init__(self, connection_string: str):
        """
        Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string
        """
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2 is required for database features. Install with: pip install psycopg2-binary")
        
        self.connection_string = connection_string
        self.conn: Optional[psycopg2.extensions.connection] = None

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            self.conn.autocommit = False
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Disconnected from database")

    def initialize_schema(self):
        """Initialize database schema with pgvector extension."""
        if not self.conn:
            self.connect()

        try:
            with self.conn.cursor() as cur:
                embedding_dimension = getattr(settings, "embedding_dimension", 384) or 384

                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Create repositories table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS repositories (
                        id SERIAL PRIMARY KEY,
                        url TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        local_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Create files table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        id SERIAL PRIMARY KEY,
                        repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
                        path TEXT NOT NULL,
                        language TEXT,
                        size_bytes INTEGER,
                        fingerprint TEXT,
                        content_hash TEXT,
                        modified_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(repository_id, path)
                    );
                """)
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_files_fingerprint ON files(repository_id, fingerprint);"
                )

                # Create chunks table with vector column
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id SERIAL PRIMARY KEY,
                        file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                        content TEXT NOT NULL,
                        start_line INTEGER,
                        end_line INTEGER,
                        start_byte INTEGER,
                        end_byte INTEGER,
                        chunk_type TEXT,
                        language TEXT,
                        metadata JSONB,
                        embedding vector({embedding_dimension}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )

                # Create commits table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS commits (
                        id SERIAL PRIMARY KEY,
                        repository_id INTEGER REFERENCES repositories(id) ON DELETE CASCADE,
                        commit_hash TEXT NOT NULL,
                        author TEXT,
                        message TEXT,
                        timestamp TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(repository_id, commit_hash)
                    );
                """)

                # Create file_commits junction table (commit lineage)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS file_commits (
                        id SERIAL PRIMARY KEY,
                        file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                        commit_id INTEGER REFERENCES commits(id) ON DELETE CASCADE,
                        change_type TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(file_id, commit_id)
                    );
                """)

                # Create relations table (graph-based relations)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS relations (
                        id SERIAL PRIMARY KEY,
                        source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                        target_chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
                        relation_type TEXT NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source_chunk_id, target_chunk_id, relation_type)
                    );
                """)

                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);")
                # IVFFlat index - lists parameter should be sqrt(total_rows) for optimal performance
                # Starting with 100, but should be adjusted based on dataset size in production
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_files_repository_id ON files(repository_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_commits_repository_id ON commits(repository_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_file_commits_file_id ON file_commits(file_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_file_commits_commit_id ON file_commits(commit_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_chunk_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_chunk_id);")

                self.conn.commit()
                logger.info("Database schema initialized")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to initialize schema: {e}")
            raise

    def insert_repository(self, url: str, name: str, local_path: Optional[str] = None) -> int:
        """Insert a repository record."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO repositories (url, name, local_path)
                VALUES (%s, %s, %s)
                ON CONFLICT (url) DO UPDATE SET
                    name = EXCLUDED.name,
                    local_path = EXCLUDED.local_path,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
                """,
                (url, name, local_path),
            )
            repo_id = cur.fetchone()[0]
            self.conn.commit()
            return repo_id

    def insert_file(
        self,
        repository_id: int,
        path: str,
        language: str,
        size_bytes: int,
        fingerprint: Optional[str] = None,
        content_hash: Optional[str] = None,
        modified_at: Optional[datetime] = None,
    ) -> int:
        """Insert a file record."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO files (repository_id, path, language, size_bytes, content_hash, modified_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (repository_id, path) DO UPDATE SET
                    language = EXCLUDED.language,
                    size_bytes = EXCLUDED.size_bytes,
                    content_hash = COALESCE(EXCLUDED.content_hash, files.content_hash),
                    modified_at = COALESCE(EXCLUDED.modified_at, files.modified_at),
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id;
                """,
                (repository_id, path, language, size_bytes, content_hash, modified_at),
            )
            file_id = cur.fetchone()[0]
            self.conn.commit()
            return file_id

    def update_file_path(self, file_id: int, new_path: str):
        """Update a file's stored path, preserving history and relations."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE files
                SET path = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s;
                """,
                (new_path, file_id),
            )
            self.conn.commit()

    def find_file_by_fingerprint(self, repository_id: int, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Locate an existing file entry by fingerprint within a repository."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, path
                FROM files
                WHERE repository_id = %s AND fingerprint = %s
                LIMIT 1;
                """,
                (repository_id, fingerprint),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"id": row[0], "path": row[1]}

    def get_file_by_path(self, repository_id: int, path: str) -> Optional[int]:
        """Fetch a file id by its repository-relative path."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id
                FROM files
                WHERE repository_id = %s AND path = %s
                LIMIT 1;
                """,
                (repository_id, path),
            )
            row = cur.fetchone()
            return row[0] if row else None
    def get_files_for_repository(self, repository_id: int) -> Dict[str, Dict[str, Any]]:
        """Fetch file metadata for a repository keyed by relative path."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, path, content_hash, modified_at
                FROM files
                WHERE repository_id = %s;
                """,
                (repository_id,),
            )

            files: Dict[str, Dict[str, Any]] = {}
            for row in cur.fetchall():
                files[row[1]] = {"id": row[0], "content_hash": row[2], "modified_at": row[3]}
            return files

    def delete_files(self, repository_id: int, paths: List[str]) -> None:
        """Delete files (and cascading chunks/relations) by path list."""
        if not paths:
            return

        with self.conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM files
                WHERE repository_id = %s AND path = ANY(%s);
                """,
                (repository_id, paths),
            )
            self.conn.commit()

    def insert_chunk(
        self,
        file_id: int,
        content: str,
        start_line: int,
        end_line: int,
        start_byte: int,
        end_byte: int,
        chunk_type: str,
        language: str,
        metadata: Dict[str, Any],
        embedding: np.ndarray,
    ) -> int:
        """Insert a chunk record with embedding."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (
                    file_id, content, start_line, end_line, start_byte, end_byte,
                    chunk_type, language, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    file_id,
                    content,
                    start_line,
                    end_line,
                    start_byte,
                    end_byte,
                    chunk_type,
                    language,
                    psycopg2.extras.Json(metadata),
                    embedding.tolist(),
                ),
            )
            chunk_id = cur.fetchone()[0]
            self.conn.commit()
            return chunk_id

    def get_chunk_ids_for_file(self, file_id: int) -> List[int]:
        """Return all chunk ids associated with a file."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM chunks WHERE file_id = %s;
                """,
                (file_id,),
            )
            return [row[0] for row in cur.fetchall()]

    def delete_chunks_for_file(self, file_id: int):
        """Remove chunks for a file, cascading dependent relations via database FKs."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE file_id = %s;", (file_id,))
            self.conn.commit()

    def delete_relations_for_chunks(self, chunk_ids: List[int]):
        """Delete relations where impacted chunks participate as source or target."""
        if not chunk_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM relations
                WHERE source_chunk_id = ANY(%s) OR target_chunk_id = ANY(%s);
                """,
                (chunk_ids, chunk_ids),
            )
            self.conn.commit()

    def get_relations_by_target_ids(self, target_chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch relations that point to provided target chunk ids."""
        if not target_chunk_ids:
            return []

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_chunk_id, target_chunk_id, relation_type, metadata
                FROM relations
                WHERE target_chunk_id = ANY(%s);
                """,
                (target_chunk_ids,),
            )
            relations: List[Dict[str, Any]] = []
            for row in cur.fetchall():
                relations.append(
                    {
                        "source_chunk_id": row[0],
                        "target_chunk_id": row[1],
                        "relation_type": row[2],
                        "metadata": row[3] or {},
                    }
                )
            return relations
    def insert_chunks_batch(self, file_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
        """Insert multiple chunk records in a single statement."""
        if not chunks:
            return []

        with self.conn.cursor() as cur:
            query = """
                INSERT INTO chunks (
                    file_id, content, start_line, end_line, start_byte, end_byte,
                    chunk_type, language, metadata, embedding
                ) VALUES %s
                RETURNING id;
            """

            template = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            data = [
                (
                    file_id,
                    c["content"],
                    c["start_line"],
                    c["end_line"],
                    c["start_byte"],
                    c["end_byte"],
                    c["chunk_type"],
                    c["language"],
                    psycopg2.extras.Json(c.get("metadata", {})),
                    c["embedding"].tolist(),
                )
                for c in chunks
            ]

            result = execute_values(cur, query, data, template=template, fetch=True)
            chunk_ids = [row[0] for row in (result or [])]
            self.conn.commit()
            return chunk_ids

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        repository_id: Optional[int] = None,
        repository_ids: Optional[List[int]] = None,
        languages: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        use_lexical: bool = False,
        lexical_weight: float = 0.3,
        query_text: Optional[str] = None,
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
        Search for similar chunks using vector similarity with optional lexical/BM25 blending.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            repository_id: Optional single repository filter (backward compatibility)
            repository_ids: Optional list of repository IDs to filter
            languages: Optional list of languages to filter
            since: Optional timestamp filter to prefer recent chunks
            use_lexical: Whether to add lexical/BM25-style scoring
            lexical_weight: Weight for lexical score when blending with vector similarity
            query_text: Raw query text used for lexical scoring
            reranker: Optional callable that returns cross-encoder scores for results
            rerank_weight: Blend factor for reranker scores vs. base combined score
            rerank_top_k: Number of top results to send to reranker (defaults to all)
            deduplicate: Drop near-identical snippets after scoring
            dedup_threshold: Similarity threshold (0-1) for deduplication
            boost_top_level: Bonus for top-level definitions (depth=0)
            boost_recent_edit: Bonus for recently edited chunks (metadata flag)
            filter_match_boost: Bonus for chunks that match language/repo filters

        Returns:
            List of matching chunks with metadata
        """
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        # Normalize repository filters for backward compatibility
        repo_filters = repository_ids or []
        if repository_id and repository_id not in repo_filters:
            repo_filters.append(repository_id)

        # Clamp lexical weight to [0, 1]
        lexical_weight = max(0.0, min(1.0, lexical_weight))
        vector_weight = 1.0 - lexical_weight
        rerank_weight = max(0.0, min(1.0, rerank_weight))

        with self.conn.cursor() as cur:
            conditions: List[str] = []
            params: List[Any] = []

            lexical_select = "NULL::float AS lexical_score"
            if use_lexical and query_text:
                lexical_select = "ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery(%s)) AS lexical_score"
                params.append(query_text)

            # Vector similarity score
            vector_score_select = "1 - (c.embedding <=> %s::vector) AS vector_score"
            params.append(query_embedding.tolist())

            if repo_filters:
                conditions.append("r.id = ANY(%s)")
                params.append(repo_filters)

            if languages:
                conditions.append("c.language = ANY(%s)")
                params.append(languages)

            if since:
                conditions.append("c.created_at >= %s")
                params.append(since)

            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)

            inner_select = f"""
                SELECT c.id, c.content, c.start_line, c.end_line, c.chunk_type,
                       c.language, c.metadata, f.path, r.name as repository_name,
                       {vector_score_select},
                       {lexical_select}
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                JOIN repositories r ON f.repository_id = r.id
                {where_clause}
            """

            order_clause = "ranked.vector_score DESC"
            if use_lexical and query_text:
                order_clause = "combined_score DESC, ranked.vector_score DESC"

            query = f"""
                SELECT ranked.*, (%s * ranked.vector_score + %s * COALESCE(ranked.lexical_score, 0)) AS combined_score
                FROM (
                    {inner_select}
                ) ranked
                ORDER BY {order_clause}
                LIMIT %s;
            """

            # Parameter order matches placeholders in query
            execute_params: List[Any] = [vector_weight, lexical_weight]
            execute_params.extend(params)
            execute_params.append(limit)

            cur.execute(query, execute_params)

            results = []
            for row in cur.fetchall():
                lexical_value = None if not use_lexical else (None if row[10] is None else float(row[10]))

                results.append({
                    "id": row[0],
                    "content": row[1],
                    "start_line": row[2],
                    "end_line": row[3],
                    "chunk_type": row[4],
                    "language": row[5],
                    "metadata": row[6],
                    "file_path": row[7],
                    "repository": row[8],
                    "similarity": float(row[9]),
                    "lexical_score": lexical_value,
                })

            # Apply blended score locally to ensure deterministic ordering for tests/fallbacks
            for result in results:
                lexical_score = result.get("lexical_score") or 0.0
                result["combined_score"] = vector_weight * result["similarity"] + lexical_weight * lexical_score

                # Boost top-level definitions or recent edits if signaled in metadata
                metadata = result.get("metadata") or {}
                depth = metadata.get("depth")
                if boost_top_level and depth == 0:
                    result["combined_score"] += boost_top_level
                    result["boost_reason"] = "top_level"

                if boost_recent_edit and metadata.get("recent_edit"):
                    result["combined_score"] += boost_recent_edit
                    result["boost_reason"] = metadata.get("boost_reason", "recent_edit")

                if filter_match_boost:
                    if languages and result.get("language") in languages:
                        result["combined_score"] += filter_match_boost
                    # Prefer chunks whose metadata knows the repository id
                    if repo_filters and metadata.get("repository_id") in repo_filters:
                        result["combined_score"] += filter_match_boost

            results.sort(key=lambda r: r["combined_score"], reverse=True)

            # Optional cross-encoder reranking on the top-k candidates
            if reranker and query_text and results:
                rerank_k = rerank_top_k or len(results)
                candidates = results[:rerank_k]
                try:
                    rerank_scores = reranker(candidates, query_text)
                    if len(rerank_scores) == len(candidates):
                        for candidate, score in zip(candidates, rerank_scores):
                            candidate["rerank_score"] = float(score)
                            candidate["combined_score"] = (
                                (1 - rerank_weight) * candidate["combined_score"]
                                + rerank_weight * float(score)
                            )
                        results.sort(key=lambda r: r["combined_score"], reverse=True)
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.debug(f"Reranker failed, falling back to base ordering: {exc}")

            # Deduplicate near-identical snippets to reduce noise
            if deduplicate:
                deduped: List[Dict[str, Any]] = []
                for result in results:
                    is_duplicate = False
                    for kept in deduped:
                        ratio = SequenceMatcher(None, kept["content"], result["content"]).ratio()
                        if ratio >= dedup_threshold:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        deduped.append(result)
                results = deduped

            results = results[:limit]

            return results

    def insert_commit(
        self,
        repository_id: int,
        commit_hash: str,
        author: str,
        message: str,
        timestamp: datetime,
    ) -> int:
        """Insert a commit record."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO commits (repository_id, commit_hash, author, message, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (repository_id, commit_hash) DO UPDATE SET
                    author = EXCLUDED.author,
                    message = EXCLUDED.message,
                    timestamp = EXCLUDED.timestamp
                RETURNING id;
                """,
                (repository_id, commit_hash, author, message, timestamp),
            )
            commit_id = cur.fetchone()[0]
            self.conn.commit()
            return commit_id

    def delete_lineage_for_files(self, file_ids: List[int]):
        """Remove lineage entries for a set of files."""
        if not file_ids:
            return
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM file_commits WHERE file_id = ANY(%s);", (file_ids,))
            self.conn.commit()

    def link_file_commit(
        self, file_id: int, commit_id: int, change_type: str = "modified"
    ) -> int:
        """Insert or update a file-to-commit lineage entry."""

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO file_commits (file_id, commit_id, change_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (file_id, commit_id) DO UPDATE SET
                    change_type = EXCLUDED.change_type
                RETURNING id;
                """,
                (file_id, commit_id, change_type),
            )

            file_commit_id = cur.fetchone()[0]
            self.conn.commit()
            return file_commit_id

    def insert_relation(
        self,
        source_chunk_id: int,
        target_chunk_id: int,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a relation between chunks."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO relations (source_chunk_id, target_chunk_id, relation_type, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_chunk_id, target_chunk_id, relation_type) DO UPDATE SET
                    metadata = EXCLUDED.metadata
                RETURNING id;
                """,
                (source_chunk_id, target_chunk_id, relation_type, psycopg2.extras.Json(metadata or {})),
            )
            relation_id = cur.fetchone()[0]
            self.conn.commit()
            return relation_id

    def get_relations_for_chunks(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch relations where either endpoint matches the provided chunk IDs."""

        if not chunk_ids:
            return []

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_chunk_id, target_chunk_id, relation_type, metadata
                FROM relations
                WHERE source_chunk_id = ANY(%s) OR target_chunk_id = ANY(%s);
                """,
                (chunk_ids, chunk_ids),
            )

            relations = []
            for row in cur.fetchall():
                relations.append(
                    {
                        "source_chunk_id": row[0],
                        "target_chunk_id": row[1],
                        "relation_type": row[2],
                        "metadata": row[3] or {},
                    }
                )

            return relations

    def get_lineage_for_chunks(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch commit lineage entries for the files associated with chunk IDs."""

        if not chunk_ids:
            return []

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    f.path AS file_path,
                    co.commit_hash,
                    co.author,
                    co.message,
                    co.timestamp,
                    fc.change_type
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                JOIN file_commits fc ON f.id = fc.file_id
                JOIN commits co ON fc.commit_id = co.id
                WHERE c.id = ANY(%s)
                ORDER BY co.timestamp DESC;
                """,
                (chunk_ids,),
            )

            lineage = []
            for row in cur.fetchall():
                timestamp = row[5]
                lineage.append(
                    {
                        "chunk_id": row[0],
                        "file_path": row[1],
                        "commit_hash": row[2],
                        "author": row[3],
                        "message": row[4],
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
                        "change_type": row[6],
                    }
                )

            return lineage
