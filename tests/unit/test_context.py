import pytest
import numpy as np
from datetime import datetime

from codehawk.context import ContextEngine
from codehawk.chunker import CodeChunk


class DummyDatabase:
    def __init__(self):
        self.relations_called = False
        self.lineage_called = False

    def get_relations_for_chunks(self, chunk_ids):
        self.relations_called = True
        return [
            {
                "source_chunk_id": 1,
                "target_chunk_id": 2,
                "relation_type": "calls",
                "metadata": {"note": "test"},
            }
        ]

    def get_lineage_for_chunks(self, chunk_ids):
        self.lineage_called = True
        return [
            {
                "chunk_id": 1,
                "file_path": "app.py",
                "commit_hash": "abc123",
                "author": "Test Author",
                "message": "Test commit",
                "timestamp": "2024-01-01T00:00:00",
                "change_type": "modified",
            }
        ]


def test_get_context_pack_includes_relations_and_lineage(monkeypatch):
    """Ensure get_context_pack enriches results with graph relations and lineage."""

    engine = ContextEngine.__new__(ContextEngine)

    mock_results = [
        {"id": 1, "content": "code A", "file_path": "app.py", "repository": "repo"},
        {"id": 2, "content": "code B", "file_path": "utils.py", "repository": "repo"},
    ]

    monkeypatch.setattr(engine, "search", lambda query, limit=5: mock_results)
    engine.db = DummyDatabase()

    context_pack = engine.get_context_pack("test query")

    assert engine.db.relations_called
    assert engine.db.lineage_called
    assert context_pack["relations"][0]["relation_type"] == "calls"
    assert context_pack["lineage"][0]["commit_hash"] == "abc123"


def test_get_context_pack_respects_include_flags(monkeypatch):
    """Relations and lineage should be omitted when flags are disabled."""

    engine = ContextEngine.__new__(ContextEngine)
    engine.db = DummyDatabase()

    mock_results = [{"id": 1, "content": "code A", "file_path": "app.py", "repository": "repo"}]
    monkeypatch.setattr(engine, "search", lambda query, limit=5: mock_results)

    context_pack = engine.get_context_pack(
        "test query", include_relations=False, include_lineage=False
    )

    assert not engine.db.relations_called
    assert not engine.db.lineage_called
    assert context_pack["relations"] == []
    assert context_pack["lineage"] == []


class DummyIndexDatabase:
    def __init__(self, existing_files):
        self.existing_files = existing_files
        self.deleted = []
        self.files_inserted = []

    def get_files_for_repository(self, repository_id):
        return self.existing_files

    def delete_files(self, repository_id, paths):
        self.deleted.extend(paths)


def test_index_directory_skips_unchanged(tmp_path, monkeypatch):
    """Unchanged files should be skipped when hashes and mtimes match."""

    engine = ContextEngine.__new__(ContextEngine)
    engine.db = DummyIndexDatabase(existing_files={})
    file_path = tmp_path / "app.py"
    file_path.write_text("print('hello world')")

    content_hash = engine._hash_content(file_path.read_bytes())
    modified_at = datetime.fromtimestamp(file_path.stat().st_mtime)
    engine.db.existing_files[str(file_path.relative_to(tmp_path))] = {
        "id": 1,
        "content_hash": content_hash,
        "modified_at": modified_at,
    }

    calls = {"indexed": 0}
    monkeypatch.setattr(engine, "_get_code_files", lambda path: [file_path])
    monkeypatch.setattr(engine, "index_file", lambda *args, **kwargs: calls.__setitem__("indexed", calls["indexed"] + 1))

    engine.index_directory(repository_id=1, repo_path=tmp_path)

    assert calls["indexed"] == 0
    assert engine.db.deleted == []


class DummyBatchEmbedder:
    def __init__(self):
        self.calls = 0

    def _enhance_chunk_text(self, chunk):
        return chunk.content

    def generate_embeddings(self, texts):
        self.calls += 1
        return [np.ones(4, dtype=np.float32) for _ in texts]


class DummyBatchDb:
    def __init__(self):
        self.inserted_chunks = []
        self.inserted_relations = 0
        self.commit_args = None
        self.linked_commits = []

    def insert_file(self, **kwargs):
        return 1

    def insert_chunks_batch(self, file_id, chunks):
        self.inserted_chunks.extend(chunks)
        return list(range(1, len(chunks) + 1))

    def insert_relation(self, **kwargs):
        self.inserted_relations += 1

    def insert_commit(
        self,
        repository_id,
        commit_hash,
        author,
        message,
        timestamp,
    ):
        self.commit_args = {
            "repository_id": repository_id,
            "commit_hash": commit_hash,
            "author": author,
            "message": message,
            "timestamp": timestamp,
        }
        return 99

    def link_file_commit(self, file_id, commit_id, change_type="modified"):
        self.linked_commits.append((file_id, commit_id, change_type))


class DummyParser:
    def _detect_language(self, file_path):
        return "python"

    def parse_file(self, file_path, language):
        return None


class DummyChunker:
    def __init__(self, chunks):
        self.chunks = chunks

    def chunk_file(self, file_path, tree, language, source_code):
        return self.chunks


class DummyGraphAnalyzer:
    def __init__(self):
        self.received_chunk_ids = []

    def analyze_imports(self, chunks, chunk_ids):
        self.received_chunk_ids.append(list(chunk_ids))
        return []

    def analyze_calls(self, chunks, chunk_ids):
        self.received_chunk_ids.append(list(chunk_ids))
        return []

    def analyze_inheritance(self, chunks, chunk_ids):
        self.received_chunk_ids.append(list(chunk_ids))
        return []


def test_index_file_batches_embeddings_and_inserts(tmp_path):
    """index_file should batch embeddings and chunk inserts."""

    file_path = tmp_path / "module.py"
    file_path.write_text("def a():\n    pass\n\nclass B:\n    pass")

    chunks = [
        CodeChunk(
            content="def a():\n    pass",
            file_path=str(file_path),
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=15,
            chunk_type="function_definition",
            language="python",
            metadata={},
        ),
        CodeChunk(
            content="class B:\n    pass",
            file_path=str(file_path),
            start_line=3,
            end_line=4,
            start_byte=16,
            end_byte=31,
            chunk_type="class_definition",
            language="python",
            metadata={},
        ),
    ]

    engine = ContextEngine.__new__(ContextEngine)
    engine.db = DummyBatchDb()
    engine.embedder = DummyBatchEmbedder()
    engine.parser = DummyParser()
    engine.chunker = DummyChunker(chunks)
    engine.graph_analyzer = DummyGraphAnalyzer()

    engine.index_file(repository_id=1, file_path=file_path, repo_path=tmp_path)

    assert engine.embedder.calls == 1
    assert len(engine.db.inserted_chunks) == 2
    assert all(len(ids) == 2 for ids in engine.graph_analyzer.received_chunk_ids)


def test_index_file_links_commit_metadata(tmp_path, monkeypatch):
    file_path = tmp_path / "module.py"
    file_path.write_text("def a():\n    return 1")

    chunks = [
        CodeChunk(
            content=file_path.read_text(),
            file_path=str(file_path),
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=len(file_path.read_text()),
            chunk_type="function_definition",
            language="python",
            metadata={},
        )
    ]

    engine = ContextEngine.__new__(ContextEngine)
    engine.db = DummyBatchDb()
    engine.embedder = DummyBatchEmbedder()
    engine.parser = DummyParser()
    engine.chunker = DummyChunker(chunks)
    engine.graph_analyzer = DummyGraphAnalyzer()

    commit_time = datetime.utcnow()
    monkeypatch.setattr(
        engine,
        "_resolve_file_commit",
        lambda repo_path, file_path: {
            "hash": "abc123",
            "author": "User",
            "message": "feat: add module",
            "timestamp": commit_time,
            "change_type": "added",
        },
    )

    engine.index_file(repository_id=7, file_path=file_path, repo_path=tmp_path)

    assert engine.db.commit_args["commit_hash"] == "abc123"
    assert engine.db.commit_args["repository_id"] == 7
    assert engine.db.linked_commits == [(1, 99, "added")]
