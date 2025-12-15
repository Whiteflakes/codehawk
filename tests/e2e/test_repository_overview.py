"""Validate passive repository overviews expose skeletonized structure."""


def test_repository_overview_includes_skeletons(populated_database, context_engine):
    overview = context_engine.get_repository_overview(populated_database)

    assert overview, "Expected repository overview to return skeleton entries"
    helper_entry = next((row for row in overview if row["file_path"].endswith("helper.py")), None)
    assert helper_entry is not None
    assert "helper_fn" in helper_entry["skeleton"], "Skeleton should include function signatures"
