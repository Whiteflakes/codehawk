"""
Demonstration script for CodeHawk architecture.

This script demonstrates the core components without requiring external dependencies.
"""

from pathlib import Path
from codehawk.parser import TreeSitterParser
from codehawk.chunker import CodeChunker, CodeChunk
from codehawk.embeddings import EmbeddingGenerator
from codehawk.graph import GraphAnalyzer
from codehawk.config import Settings


def demo_parser():
    """Demonstrate the tree-sitter parser."""
    print("\n" + "=" * 60)
    print("1. TREE-SITTER PARSER DEMO")
    print("=" * 60)
    
    parser = TreeSitterParser()
    print(f"Available parsers: {list(parser.parsers.keys())}")
    
    # Test language detection
    test_files = [
        Path("example.py"),
        Path("example.js"),
        Path("example.ts"),
        Path("example.java"),
    ]
    
    print("\nLanguage detection:")
    for file_path in test_files:
        language = parser._detect_language(file_path)
        print(f"  {file_path.name} -> {language}")


def demo_chunker():
    """Demonstrate the code chunker."""
    print("\n" + "=" * 60)
    print("2. CODE CHUNKER DEMO")
    print("=" * 60)
    
    chunker = CodeChunker(chunk_size=100, overlap=10)
    print(f"Chunk size: {chunker.chunk_size}, Overlap: {chunker.overlap}")
    
    # Sample Python code
    sample_code = """
def calculate_sum(a, b):
    \"\"\"Calculate sum of two numbers.\"\"\"
    return a + b

def calculate_product(a, b):
    \"\"\"Calculate product of two numbers.\"\"\"
    return a * b

class Calculator:
    \"\"\"Simple calculator class.\"\"\"
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y
"""
    
    chunks = chunker._chunk_by_lines(
        file_path=Path("example.py"),
        language="python",
        source_code=sample_code,
    )
    
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n  Chunk {i}:")
        print(f"    Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"    Type: {chunk.chunk_type}")
        print(f"    Size: {len(chunk.content)} chars")
        print(f"    Preview: {chunk.content[:50]}...")


def demo_embeddings():
    """Demonstrate the embedding generator."""
    print("\n" + "=" * 60)
    print("3. EMBEDDING GENERATOR DEMO")
    print("=" * 60)
    
    embedder = EmbeddingGenerator()
    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")
    
    # Generate sample embeddings
    texts = [
        "def hello_world():",
        "function hello() {}",
        "class MyClass:",
    ]
    
    print("\nGenerating embeddings...")
    embeddings = embedder.generate_embeddings(texts)
    
    for i, (text, emb) in enumerate(zip(texts, embeddings), 1):
        print(f"\n  Text {i}: {text}")
        print(f"    Embedding shape: {emb.shape}")
        print(f"    First 5 values: {emb[:5]}")


def demo_graph_analyzer():
    """Demonstrate the graph analyzer."""
    print("\n" + "=" * 60)
    print("4. GRAPH ANALYZER DEMO")
    print("=" * 60)
    
    analyzer = GraphAnalyzer()
    
    # Sample code chunks
    chunks = [
        CodeChunk(
            content="import math\n\ndef calculate(x):\n    return math.sqrt(x)",
            file_path="math_utils.py",
            start_line=1,
            end_line=4,
            start_byte=0,
            end_byte=100,
            chunk_type="function_definition",
            language="python",
            metadata={"name": "calculate"},
        ),
        CodeChunk(
            content="def sqrt(n):\n    return n ** 0.5",
            file_path="custom_math.py",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=50,
            chunk_type="function_definition",
            language="python",
            metadata={"name": "sqrt"},
        ),
    ]
    
    chunk_ids = [1, 2]
    
    # Analyze relationships
    print("\nAnalyzing code relationships...")
    imports = analyzer.analyze_imports(chunks, chunk_ids)
    calls = analyzer.analyze_calls(chunks, chunk_ids)
    
    print(f"  Found {len(imports)} import relations")
    print(f"  Found {len(calls)} call relations")
    
    for relation in imports:
        print(f"\n  Import relation:")
        print(f"    Source: {relation.source_chunk_id}")
        print(f"    Target: {relation.target_chunk_id}")
        print(f"    Type: {relation.relation_type}")


def demo_config():
    """Demonstrate configuration management."""
    print("\n" + "=" * 60)
    print("5. CONFIGURATION DEMO")
    print("=" * 60)
    
    settings = Settings()
    
    print("\nDatabase settings:")
    print(f"  Host: {settings.db_host}")
    print(f"  Port: {settings.db_port}")
    print(f"  Database: {settings.db_name}")
    
    print("\nEmbedding settings:")
    print(f"  Model: {settings.embedding_model}")
    print(f"  Dimension: {settings.embedding_dimension}")
    print(f"  Chunk size: {settings.chunk_size}")
    
    print("\nSupported languages:")
    for lang in settings.supported_languages:
        print(f"  - {lang}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "CodeHawk Architecture Demo" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    
    demo_parser()
    demo_chunker()
    demo_embeddings()
    demo_graph_analyzer()
    demo_config()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nCodeHawk components working correctly!")
    print("\nNext steps:")
    print("  1. Install PostgreSQL with pgvector")
    print("  2. Run: codehawk init-db")
    print("  3. Index a repository: codehawk index /path/to/repo")
    print("  4. Search code: codehawk search 'your query'")
    print("\n")


if __name__ == "__main__":
    main()
