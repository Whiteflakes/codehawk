# Release readiness summary

This document tracks the current readiness of CodeHawk for release.

## Current state
- ✅ Core indexing, parsing, chunking, embedding, relation analysis, and lineage tracking paths are functional.
- ✅ Unit test suite passes (`pytest`), and end-to-end fixtures will run automatically when `testcontainers`/Docker are available.
- ✅ Embedding fallback logic and tree-sitter stubs allow offline/local development without crashes when models or native bindings are missing.
- ⚠️ Deployments still depend on external services (PostgreSQL with pgvector, tree-sitter grammars, embedding model downloads) and are not bundled by default.

## Open limitations
- Tree-sitter grammars must be installed separately for each language; missing grammars fall back to line-based chunking.
- Embedding downloads may require network access; offline mode is deterministic but suitable primarily for testing.
- Search quality is based on vector/BM25 blending without rerankers.
- API/MCP usage assumes a running PostgreSQL with pgvector and initialized schema.

## Release checklist
- [x] Run unit tests: `pytest`
- [ ] Provision PostgreSQL with pgvector and set `CODEHAWK_DB_*` environment variables
- [ ] Install required tree-sitter grammars for targeted languages
- [ ] Ensure embedding model availability or enable offline deterministic mode
- [ ] Initialize schema via `codehawk init-db` and smoke test indexing a sample repo
- [ ] (Optional) Run e2e tests with Docker: `pytest tests/e2e` (requires `testcontainers`)

## References
- Usage, API, and CLI examples: [README](README.md)
- Configuration defaults: `codehawk/config.py`
