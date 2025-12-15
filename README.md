# CodeHawk 🦅

Open-source code context engine using tree-sitter parsing, chunk-level embeddings, graph-based relations, and commit lineage to deliver high-quality context packs for LLMs.

## Features

- **🌳 Tree-sitter Parsing**: Multi-language code parsing with tree-sitter for accurate syntax analysis
- **🧩 Smart Chunking**: Semantic-aware code chunking that respects function and class boundaries
- **🔢 Embeddings**: Chunk-level embeddings using sentence-transformers for semantic search
- **🗄️ Vector Storage**: PostgreSQL + pgvector for efficient similarity search
- **📊 Graph Relations**: Analyzes imports, function calls, and inheritance relationships
- **🕰️ Commit Lineage**: Tracks file history and co-change patterns using Git
- **🔍 Search API**: REST API for code search and context retrieval
- **🔌 MCP Server**: Model Context Protocol server for LLM integration
- **⚡ Fast & Hackable**: Built in Python with uv for local development

## Supported Languages

- Python (`.py`)
- JavaScript (`.js`, `.jsx`)
- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- Go (`.go`)
- Rust (`.rs`)

To enable parsing for each language, install the corresponding tree-sitter grammar packages:

- `tree_sitter_python`
- `tree_sitter_javascript`
- `tree_sitter_typescript`
- `tree_sitter_java`
- `tree_sitter_go`
- `tree_sitter_rust`

## Installation

### Using pip

```bash
pip install -e .
```

### Using uv (recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

### Development and testing setup

Install the development requirements before running the test suite:

```bash
# Using pip
pip install -r requirements-dev.txt

# Using uv
uv pip install -r requirements-dev.txt
```

## Quick Start

### 1. Setup Database

CodeHawk requires PostgreSQL with the pgvector extension:

```bash
# Install PostgreSQL and pgvector
# On Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Create database
createdb codehawk

# Initialize schema
codehawk init-db
```

### 2. Index a Repository

```bash
# Index a local repository
codehawk index /path/to/repository --url https://github.com/user/repo

# Index with custom database URL
codehawk index /path/to/repository --db-url postgresql://user:pass@localhost/codehawk
```

### 3. Search Code

```bash
# Search for code chunks
codehawk search "function to parse JSON"

# Limit results
codehawk search "authentication middleware" --limit 5

# Filter by repository
codehawk search "database connection" --repository-id 1
```

### 4. Generate Context Pack

```bash
# Generate context pack for LLM
codehawk context "how does the authentication work?" --limit 10

# Save to file
codehawk context "error handling" --output context.json
```

### 5. Start API Server

```bash
# Start REST API
codehawk serve --host 0.0.0.0 --port 8000

# Start MCP server
codehawk mcp --host 0.0.0.0 --port 8001
```

## API Usage

### REST API

```python
import requests

# Search for code
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "authentication function",
        "limit": 10
    }
)
results = response.json()

# Get context pack
response = requests.post(
    "http://localhost:8000/context",
    json={
        "query": "how does caching work?",
        "limit": 5,
        "include_relations": True,
        "include_lineage": True
    }
)
context = response.json()
```

### Python SDK

```python
from codehawk import ContextEngine

# Initialize engine
engine = ContextEngine()
engine.initialize()

# Index a repository
repo_id = engine.index_repository("/path/to/repo")

# Search
results = engine.search("authentication", limit=10)

# Get context pack
context_pack = engine.get_context_pack(
    query="how does the API work?",
    limit=5,
    include_relations=True
)

# Cleanup
engine.shutdown()
```

### MCP Server (WebSocket)

```python
import asyncio
import websockets
import json

async def query_mcp():
    uri = "ws://localhost:8001/mcp"
    async with websockets.connect(uri) as websocket:
        # Search request
        request = {
            "method": "search",
            "params": {
                "query": "authentication",
                "limit": 5
            },
            "id": "1"
        }
        
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(query_mcp())
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Database
CODEHAWK_DB_HOST=localhost
CODEHAWK_DB_PORT=5432
CODEHAWK_DB_NAME=codehawk
CODEHAWK_DB_USER=postgres
CODEHAWK_DB_PASSWORD=postgres

# Embeddings
CODEHAWK_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CODEHAWK_EMBEDDING_DIMENSION=384
CODEHAWK_CHUNK_SIZE=512
CODEHAWK_CHUNK_OVERLAP=50

# API
CODEHAWK_API_HOST=0.0.0.0
CODEHAWK_API_PORT=8000

# MCP Server
CODEHAWK_MCP_HOST=0.0.0.0
CODEHAWK_MCP_PORT=8001

# Indexing
CODEHAWK_MAX_FILE_SIZE=1000000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CodeHawk                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Tree-sitter  │  │ Code Chunker │  │  Embeddings  │    │
│  │   Parser     │─▶│              │─▶│  Generator   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                                     │            │
│         ▼                                     ▼            │
│  ┌──────────────┐                    ┌──────────────┐    │
│  │    Graph     │                    │  PostgreSQL  │    │
│  │   Analyzer   │───────────────────▶│  + pgvector  │    │
│  └──────────────┘                    └──────────────┘    │
│         │                                     ▲            │
│         ▼                                     │            │
│  ┌──────────────┐                            │            │
│  │   Lineage    │────────────────────────────┘            │
│  │   Tracker    │                                         │
│  └──────────────┘                                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                        Interfaces                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   CLI    │  │ REST API │  │   MCP    │  │  Python  │  │
│  │          │  │          │  │  Server  │  │   SDK    │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/Whiteflakes/codehawk.git
cd codehawk

# Install dependencies with dev extras
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black codehawk tests

# Lint code
ruff check codehawk tests

# Type check
mypy codehawk
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=codehawk --cov-report=html

# Run specific test file
pytest tests/unit/test_parser.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Tree-sitter for parsing
- pgvector for vector similarity search
- sentence-transformers for embeddings
- FastAPI for API framework
