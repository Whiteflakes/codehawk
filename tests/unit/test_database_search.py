from datetime import datetime
import numpy as np
import time

from codehawk.database import Database


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.last_cursor: FakeCursor | None = None

    def cursor(self):
        self.last_cursor = FakeCursor(self.rows)
        return self.last_cursor


def test_search_chunks_hybrid_filters_and_ordering():
    rows = [
        (1, "foo", "sig foo", 1, 5, "code", "python", {}, "file1.py", "repo", 0.70, 0.20),
        (2, "bar", "sig bar", 6, 9, "code", "python", {}, "file2.py", "repo", 0.40, 0.90),
    ]

    db = Database.__new__(Database)
    db.conn = FakeConnection(rows)

    results = db.search_chunks(
        np.array([0.1, 0.2, 0.3]),
        limit=2,
        repository_ids=[1],
        languages=["python"],
        since=datetime(2024, 1, 1),
        use_lexical=True,
        lexical_weight=0.6,
        query_text="foo bar",
    )

    assert results[0]["id"] == 2  # lexical match dominates after blending
    assert results[0]["combined_score"] > results[1]["combined_score"]

    executed_query, params = db.conn.last_cursor.executed[0]
    assert "r.id = ANY" in executed_query
    assert "c.language = ANY" in executed_query
    assert "c.created_at >=" in executed_query
    assert params[4] == [1]  # repository filter parameter
    assert params[5] == ["python"]


def reciprocal_rank(ranking, relevant_id):
    for idx, value in enumerate(ranking, start=1):
        if value == relevant_id:
            return 1.0 / idx
    return 0.0


def test_hybrid_search_improves_mrr_and_stays_fast():
    rows = [
        (1, "foo", "sig foo", 1, 5, "code", "python", {}, "file1.py", "repo", 0.70, 0.10),
        (2, "bar", "sig bar", 6, 9, "code", "python", {}, "file2.py", "repo", 0.40, 0.90),
    ]

    db = Database.__new__(Database)
    db.conn = FakeConnection(rows)

    start = time.perf_counter()
    vector_results = db.search_chunks(np.array([0.1, 0.2, 0.3]), limit=2, use_lexical=False)
    vector_latency = time.perf_counter() - start

    start = time.perf_counter()
    hybrid_results = db.search_chunks(
        np.array([0.1, 0.2, 0.3]),
        limit=2,
        use_lexical=True,
        lexical_weight=0.6,
        query_text="bar",  # lexical signal should surface id=2 first
    )
    hybrid_latency = time.perf_counter() - start

    vector_mrr = reciprocal_rank([r["id"] for r in vector_results], 2)
    hybrid_mrr = reciprocal_rank([r["id"] for r in hybrid_results], 2)

    assert hybrid_mrr > vector_mrr
    # Hybrid path should remain within 2x the vector-only latency for small result sets
    assert hybrid_latency <= vector_latency * 2 + 0.001


def test_reranker_and_deduplication_boost_mrr_without_huge_latency():
    rows = [
        (
            1,
            "def helper():\n    return 1",
            "def helper(): ...",
            1,
            3,
            "function_definition",
            "python",
            {"depth": 1},
            "helpers.py",
            "repoA",
            0.85,
            0.10,
        ),
        (
            2,
            "def helper():\n    return 1",
            "def helper(): ...",
            10,
            12,
            "function_definition",
            "python",
            {"depth": 0, "recent_edit": True, "repository_id": 42},
            "helpers.py",
            "repoA",
            0.80,
            0.05,
        ),
        (
            3,
            "class Service:\n    pass",
            "class Service: ...",
            20,
            25,
            "class_definition",
            "python",
            {"depth": 0},
            "service.py",
            "repoB",
            0.40,
            0.60,
        ),
    ]

    db = Database.__new__(Database)
    db.conn = FakeConnection(rows)

    def fake_reranker(results, query):
        # Pretend the cross-encoder strongly prefers the recent edit (id=2)
        scores = []
        for result in results:
            if result["id"] == 2:
                scores.append(0.95)
            elif result["id"] == 3:
                scores.append(0.2)
            else:
                scores.append(0.1)
        return scores

    start = time.perf_counter()
    base_results = db.search_chunks(
        np.array([0.1, 0.2, 0.3]),
        limit=3,
        use_lexical=False,
        deduplicate=False,
        boost_top_level=0.0,
        boost_recent_edit=0.0,
        filter_match_boost=0.0,
    )
    base_latency = time.perf_counter() - start

    start = time.perf_counter()
    reranked_results = db.search_chunks(
        np.array([0.1, 0.2, 0.3]),
        limit=3,
        use_lexical=True,
        lexical_weight=0.3,
        query_text="recent helper",
        repository_ids=[42],
        languages=["python"],
        reranker=fake_reranker,
        rerank_weight=0.6,
        rerank_top_k=3,
        deduplicate=True,
        dedup_threshold=0.9,
        boost_top_level=0.1,
        boost_recent_edit=0.1,
        filter_match_boost=0.05,
    )
    rerank_latency = time.perf_counter() - start

    base_mrr = reciprocal_rank([r["id"] for r in base_results], 2)
    rerank_mrr = reciprocal_rank([r["id"] for r in reranked_results], 2)

    assert rerank_mrr > base_mrr
    assert reranked_results[0]["id"] == 2
    # Deduplication should merge the two helper snippets
    assert len(reranked_results) == 2
    # Reranking path should be fast enough for small candidate sets
    assert rerank_latency <= base_latency * 3 + 0.001
