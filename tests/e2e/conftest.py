"""Shared fixtures for end-to-end tests."""

import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from testcontainers.postgres import PostgresContainer

from codehawk.context import ContextEngine
from codehawk.embeddings import EmbeddingGenerator, DeterministicEmbeddingBackend
from codehawk.config import settings


@pytest.fixture(scope="session")
def postgres_url():
    """Spin up a temporary PostgreSQL instance with pgvector enabled."""

    with PostgresContainer("ankane/pgvector:latest", driver="psycopg2") as pg:
        connection_url = pg.get_connection_url()
        # Normalize URL format for psycopg2
        yield connection_url.replace("postgresql+psycopg2", "postgresql")


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory) -> Path:
    """Create a small git repository with simple Python relations."""

    repo_path = tmp_path_factory.mktemp("sample_repo")

    (repo_path / "helper.py").write_text(
        """
def helper_fn():
    return "hello"


class Greeter:
    def greet(self):
        return helper_fn()
""".strip()
    )

    (repo_path / "main.py").write_text(
        """
from helper import helper_fn


def call_helper():
    return helper_fn()


if __name__ == "__main__":
    print(call_helper())
""".strip()
    )

    subprocess.run(["git", "init"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo_path, check=True)
    subprocess.run(["git", "add", "helper.py", "main.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True)

    # A follow-up change to produce distinct lineage entries
    (repo_path / "helper.py").write_text(
        """
def helper_fn():
    return "hello"


def helper_with_arg(name: str) -> str:
    return f"Hi {name}"


class Greeter:
    def greet(self):
        return helper_fn()
""".strip()
    )

    subprocess.run(["git", "add", "helper.py"], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "add helper overload"], cwd=repo_path, check=True)

    return repo_path


@pytest.fixture
def context_engine(postgres_url, monkeypatch):
    """Provision a ContextEngine backed by the temporary database."""

    monkeypatch.setenv("CODEHAWK_EMBEDDING_OFFLINE_MODE", "1")
    monkeypatch.setenv("CODEHAWK_DISABLE_MODEL_LOAD", "1")

    engine = ContextEngine(database_url=postgres_url)
    engine.embedder = EmbeddingGenerator(
        backend=DeterministicEmbeddingBackend(settings.embedding_dimension),
        offline_mode=True,
    )

    engine.initialize()

    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture
def populated_database(context_engine, sample_repo):
    """Index the sample repository and attach lineage info."""

    repo_id = context_engine.index_repository(sample_repo)
    db = context_engine.db

    assert db is not None

    # Ensure at least one commit lineage entry exists for each file
    commit_id = db.insert_commit(
        repository_id=repo_id,
        commit_hash="integration-test-commit",
        author="CI",
        message="integration lineage",
        timestamp=datetime.utcnow(),
    )

    for file_meta in db.get_files_for_repository(repo_id).values():
        db.link_file_commit(file_meta["id"], commit_id, change_type="modified")

    return repo_id
