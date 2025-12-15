from codehawk.chunker import CodeChunk
from codehawk.graph import GraphAnalyzer


def test_graph_analyzer_links_cross_file_symbols():
    analyzer = GraphAnalyzer()

    app_chunk = CodeChunk(
        content="from utils import helper\n\ndef run():\n    helper()",
        file_path="repo/app.py",
        start_line=1,
        end_line=4,
        start_byte=0,
        end_byte=0,
        chunk_type="function_definition",
        language="python",
        metadata={},
    )

    helper_chunk = CodeChunk(
        content="def helper():\n    return True",
        file_path="repo/utils.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=0,
        chunk_type="function_definition",
        language="python",
        metadata={},
    )

    base_chunk = CodeChunk(
        content="class Base:\n    pass",
        file_path="repo/base.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=0,
        chunk_type="class_definition",
        language="python",
        metadata={},
    )

    derived_chunk = CodeChunk(
        content="from base import Base\n\nclass Derived(Base):\n    pass",
        file_path="repo/models.py",
        start_line=1,
        end_line=4,
        start_byte=0,
        end_byte=0,
        chunk_type="class_definition",
        language="python",
        metadata={},
    )

    chunks = [app_chunk, helper_chunk, base_chunk, derived_chunk]
    chunk_ids = [1, 2, 3, 4]

    import_relations = analyzer.analyze_imports(chunks, chunk_ids)
    import_edges = {(rel.source_chunk_id, rel.target_chunk_id) for rel in import_relations}
    assert (1, 2) in import_edges
    assert (4, 3) in import_edges

    call_relations = analyzer.analyze_calls(chunks, chunk_ids)
    call_edges = {(rel.source_chunk_id, rel.target_chunk_id) for rel in call_relations}
    assert (1, 2) in call_edges

    inheritance_relations = analyzer.analyze_inheritance(chunks, chunk_ids)
    inheritance_edges = {(rel.source_chunk_id, rel.target_chunk_id) for rel in inheritance_relations}
    assert (4, 3) in inheritance_edges
