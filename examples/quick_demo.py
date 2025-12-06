"""
Quick demo of CodeHawk core features without heavy dependencies.
"""

from pathlib import Path
from codehawk.parser import TreeSitterParser
from codehawk.chunker import CodeChunker
from codehawk.config import Settings


def main():
    """Run quick demo."""
    print("\n" + "=" * 60)
    print("CodeHawk - Quick Demo")
    print("=" * 60)
    
    # 1. Configuration
    print("\n1. Configuration")
    settings = Settings()
    print(f"   Database: {settings.db_name}")
    print(f"   Chunk size: {settings.chunk_size}")
    print(f"   Supported languages: {', '.join(settings.supported_languages)}")
    
    # 2. Parser
    print("\n2. Parser")
    parser = TreeSitterParser()
    print(f"   Initialized with {len(parser.parsers)} parser(s)")
    print(f"   Language detection:")
    for ext, lang in [(".py", "python"), (".js", "javascript"), (".ts", "typescript")]:
        detected = parser._detect_language(Path(f"test{ext}"))
        print(f"     {ext} -> {detected}")
    
    # 3. Chunker
    print("\n3. Chunker")
    chunker = CodeChunker(chunk_size=200, overlap=20)
    
    sample_code = """def hello():
    print("Hello")

def world():
    print("World")

class Example:
    def method(self):
        pass
"""
    
    chunks = chunker._chunk_by_lines(
        file_path=Path("example.py"),
        language="python",
        source_code=sample_code,
    )
    
    print(f"   Generated {len(chunks)} chunks from sample code")
    for i, chunk in enumerate(chunks, 1):
        print(f"     Chunk {i}: lines {chunk.start_line}-{chunk.end_line}, {len(chunk.content)} chars")
    
    print("\n" + "=" * 60)
    print("Demo complete! All core components working.")
    print("\nTo use full features:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Setup PostgreSQL with pgvector")
    print("  3. Run: codehawk init-db")
    print("  4. Index code: codehawk index /path/to/repo")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
