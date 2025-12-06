"""
Example usage of CodeHawk context engine.

This example demonstrates how to:
1. Initialize the context engine
2. Index a repository
3. Search for code
4. Generate context packs
"""

from pathlib import Path
from codehawk import ContextEngine


def main():
    """Run example usage."""
    print("CodeHawk Example\n" + "=" * 50)
    
    # Initialize engine
    print("\n1. Initializing context engine...")
    engine = ContextEngine()
    
    # Note: This requires a PostgreSQL database with pgvector extension
    # For this example, we'll handle the error gracefully
    try:
        engine.initialize()
        print("   ✓ Engine initialized")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Note: Please ensure PostgreSQL with pgvector is installed")
        print("   and configure database settings in .env or environment variables")
        return
    
    # Index a repository
    print("\n2. Indexing repository...")
    try:
        repo_path = Path(".")  # Current directory
        repo_id = engine.index_repository(repo_path)
        print(f"   ✓ Indexed repository (ID: {repo_id})")
    except Exception as e:
        print(f"   ✗ Error indexing: {e}")
        engine.shutdown()
        return
    
    # Search for code
    print("\n3. Searching for code...")
    try:
        results = engine.search("function definition", limit=5)
        print(f"   ✓ Found {len(results)} results")
        
        for i, result in enumerate(results[:3], 1):
            print(f"\n   Result {i}:")
            print(f"   - File: {result['file_path']}")
            print(f"   - Lines: {result['start_line']}-{result['end_line']}")
            print(f"   - Type: {result['chunk_type']}")
            print(f"   - Similarity: {result['similarity']:.4f}")
    except Exception as e:
        print(f"   ✗ Error searching: {e}")
    
    # Generate context pack
    print("\n4. Generating context pack...")
    try:
        context_pack = engine.get_context_pack(
            query="how does the parser work?",
            limit=3,
        )
        print(f"   ✓ Generated context pack with {len(context_pack['chunks'])} chunks")
        
        # Show context pack structure
        print("\n   Context pack structure:")
        print(f"   - Query: {context_pack['query']}")
        print(f"   - Chunks: {len(context_pack['chunks'])}")
        print(f"   - Relations: {len(context_pack['relations'])}")
        print(f"   - Lineage: {len(context_pack['lineage'])}")
    except Exception as e:
        print(f"   ✗ Error generating context: {e}")
    
    # Cleanup
    print("\n5. Shutting down...")
    engine.shutdown()
    print("   ✓ Engine shutdown complete")
    
    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == "__main__":
    main()
