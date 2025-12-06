"""Command-line interface for CodeHawk."""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from codehawk.context import ContextEngine
from codehawk.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """CodeHawk - Open-source code context engine."""
    pass


@main.command()
@click.argument("repository_path", type=click.Path(exists=True, file_okay=False))
@click.option("--url", help="Repository URL")
@click.option("--db-url", help="Database connection URL", default=None)
def index(repository_path: str, url: Optional[str], db_url: Optional[str]):
    """Index a repository for code search."""
    click.echo(f"Indexing repository: {repository_path}")
    
    try:
        engine = ContextEngine(database_url=db_url)
        engine.initialize()
        
        repo_path = Path(repository_path)
        repository_id = engine.index_repository(repo_path, url)
        
        click.echo(f"✓ Successfully indexed repository (ID: {repository_id})")
        
        engine.shutdown()
    except Exception as e:
        click.echo(f"✗ Error indexing repository: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option("--limit", default=10, help="Maximum number of results")
@click.option("--repository-id", type=int, help="Filter by repository ID")
@click.option("--db-url", help="Database connection URL", default=None)
def search(query: str, limit: int, repository_id: Optional[int], db_url: Optional[str]):
    """Search for code chunks."""
    click.echo(f"Searching for: {query}")
    
    try:
        engine = ContextEngine(database_url=db_url)
        engine.initialize()
        
        results = engine.search(query, limit=limit, repository_id=repository_id)
        
        if not results:
            click.echo("No results found.")
        else:
            click.echo(f"\nFound {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. {result['file_path']} (lines {result['start_line']}-{result['end_line']})")
                click.echo(f"   Repository: {result['repository']}")
                click.echo(f"   Type: {result['chunk_type']} | Language: {result['language']}")
                click.echo(f"   Similarity: {result['similarity']:.4f}")
                click.echo(f"\n   {result['content'][:200]}...")
                click.echo()
        
        engine.shutdown()
    except Exception as e:
        click.echo(f"✗ Error searching: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of chunks to include")
@click.option("--db-url", help="Database connection URL", default=None)
@click.option("--output", type=click.File("w"), default="-", help="Output file (default: stdout)")
def context(query: str, limit: int, db_url: Optional[str], output):
    """Generate a context pack for LLM."""
    try:
        engine = ContextEngine(database_url=db_url)
        engine.initialize()
        
        context_pack = engine.get_context_pack(query, limit=limit)
        
        import json
        json.dump(context_pack, output, indent=2)
        
        if output.name != "<stdout>":
            click.echo(f"✓ Context pack written to {output.name}")
        
        engine.shutdown()
    except Exception as e:
        click.echo(f"✗ Error generating context: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
def serve(host: str, port: int):
    """Start the API server."""
    click.echo(f"Starting CodeHawk API server on {host}:{port}")
    
    try:
        import uvicorn
        from codehawk.api import app
        
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        click.echo(f"✗ Error starting server: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="MCP server host")
@click.option("--port", default=8001, help="MCP server port")
def mcp(host: str, port: int):
    """Start the MCP server."""
    click.echo(f"Starting CodeHawk MCP server on {host}:{port}")
    
    try:
        import uvicorn
        from codehawk.mcp import app
        
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        click.echo(f"✗ Error starting MCP server: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--db-url", help="Database connection URL", default=None)
def init_db(db_url: Optional[str]):
    """Initialize the database schema."""
    click.echo("Initializing database schema...")
    
    try:
        from codehawk.database import Database
        
        db = Database(db_url or settings.database_url)
        db.connect()
        db.initialize_schema()
        db.disconnect()
        
        click.echo("✓ Database schema initialized")
    except Exception as e:
        click.echo(f"✗ Error initializing database: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
