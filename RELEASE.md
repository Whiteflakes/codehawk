# CodeHawk Release Checklist

Use this checklist to ensure CodeHawk is production-ready before publishing a release or deploying to a new environment.

## 1. Environment preparation
- [ ] Install system dependencies: PostgreSQL + pgvector extension, Git, and build tools for tree-sitter grammars.
- [ ] Install Python dependencies with dev extras: `pip install -e .[dev]`.
- [ ] Install target tree-sitter grammars (Python, JavaScript/TypeScript, Java, Go, Rust). Missing grammars fall back to line-based chunking but reduce accuracy.
- [ ] Confirm embedding model availability or set offline deterministic mode via environment variables.

## 2. Database and storage
- [ ] Provision PostgreSQL with pgvector and set `CODEHAWK_DB_*` env vars (host, port, name, user, password).
- [ ] Run `codehawk init-db` to create tables and vector indexes.
- [ ] (Optional) Configure separate databases for testing and production.

## 3. Validation
- [ ] Run unit tests: `pytest` (already passing in this workspace).
- [ ] Install `testcontainers` and run end-to-end tests: `pytest tests/e2e` (requires Docker and pgvector image). Verify the suite can pull images and run containers.
- [ ] Smoke test indexing: `codehawk index <repo_path> --url <repo_url>` and ensure chunks and embeddings are written to the database.
- [ ] Smoke test search and context pack generation:
  - `codehawk search "sample query" --limit 5 --use-lexical`
  - `codehawk context "sample question" --limit 5`
- [ ] Exercise API and MCP servers: `codehawk serve` and `codehawk mcp`, then issue a search/context request against each service.

## 4. Documentation and metadata
- [ ] Verify README examples against the current CLI/API.
- [ ] Update `STATUS.md` with the latest results and note any environment-specific caveats.
- [ ] Tag the release in version control and ensure `pyproject.toml`/`codehawk.__version__` match.

## 5. Operational notes
- End-to-end behavior depends on external services; bundle infra-as-code or deployment scripts as needed for your target environment.
- Offline embedding mode is deterministic and intended for testing; prefer real embeddings for production search quality.
- If Docker is unavailable, rely on unit tests and manual smoke tests against a provisioned database.
