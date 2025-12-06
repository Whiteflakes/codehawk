import os
import tempfile
from pathlib import Path

import git

from codehawk.graph import GraphUpdatePlanner, CodeRelation
from codehawk.chunker import CodeChunk
from codehawk.lineage import LineageTracker


def _init_repo(tmpdir: Path) -> git.Repo:
    repo = git.Repo.init(tmpdir)
    repo.index.commit("initial")
    return repo


def test_detects_rename_and_fingerprint_match():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        repo = _init_repo(repo_path)

        original = repo_path / "module.py"
        original.write_text("print('hello')\n")
        repo.index.add([str(original)])
        first_commit = repo.index.commit("add module")

        renamed = repo_path / "renamed_module.py"
        os.rename(original, renamed)
        repo.index.add([str(renamed)])
        repo.index.remove([str(original)])
        second_commit = repo.index.commit("rename module")

        tracker = LineageTracker(repo_path)
        tracker.open_repository()

        changes = tracker.detect_changes(first_commit.hexsha, second_commit.hexsha)

        assert any(change.change_type == "renamed" for change in changes)
        rename = next(change for change in changes if change.change_type == "renamed")
        assert rename.previous_path == "module.py"

        fingerprint_before = tracker.get_blob_fingerprint(original, first_commit.hexsha)
        assert rename.fingerprint == fingerprint_before


def test_impacted_chunks_include_ancestors():
    chunk_a = CodeChunk(
        content="import lib",
        file_path="a.py",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=10,
        chunk_type="function_definition",
        language="python",
        metadata={},
    )
    chunk_b = CodeChunk(
        content="def lib(): pass",
        file_path="lib.py",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=15,
        chunk_type="function_definition",
        language="python",
        metadata={},
    )

    relations = [
        CodeRelation(
            source_chunk_id=1,
            target_chunk_id=2,
            relation_type="imports",
            metadata={"source_file": chunk_a.file_path, "target_file": chunk_b.file_path},
        )
    ]

    planner = GraphUpdatePlanner(None)
    impacted = planner.impacted_chunks(relations, {2})

    assert impacted == {1, 2}
