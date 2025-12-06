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
        (1, "foo", 1, 5, "code", "python", {}, "file1.py", "repo", 0.70, 0.20),
        (2, "bar", 6, 9, "code", "python", {}, "file2.py", "repo", 0.40, 0.90),
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
        (1, "foo", 1, 5, "code", "python", {}, "file1.py", "repo", 0.70, 0.10),
        (2, "bar", 6, 9, "code", "python", {}, "file2.py", "repo", 0.40, 0.90),
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
