# Release readiness summary

This document tracks the current readiness of CodeHawk for release and records the latest verification steps.

## Current state
- ✅ Core indexing, parsing, chunking, embedding, relation analysis, and lineage tracking paths are functional.
- ✅ Unit test suite passes (`pytest`), and end-to-end fixtures will run automatically when `testcontainers`/Docker are available.
- ✅ Embedding fallback logic and tree-sitter stubs allow offline/local development without crashes when models or native bindings are missing.
- ⚠️ Deployments still depend on external services (PostgreSQL with pgvector, tree-sitter grammars, embedding model downloads) and are not bundled by default.
- ⚠️ End-to-end tests are skipped in this environment because `testcontainers`/Docker are not present; run them in a Docker-capable environment before publishing.

## Open limitations
- Tree-sitter grammars must be installed separately for each language; missing grammars fall back to line-based chunking.
- Embedding downloads may require network access; offline mode is deterministic but suitable primarily for testing.
- Search quality is based on vector/BM25 blending without rerankers.
- API/MCP usage assumes a running PostgreSQL with pgvector and initialized schema.

## Release checklist
- [x] Run unit tests: `pytest`
- [ ] Install `testcontainers` and run end-to-end suite: `pytest tests/e2e` (requires Docker and pgvector image)
- [ ] Provision PostgreSQL with pgvector and set `CODEHAWK_DB_*` environment variables
- [ ] Install required tree-sitter grammars for targeted languages
- [ ] Ensure embedding model availability or enable offline deterministic mode
- [ ] Initialize schema via `codehawk init-db` and smoke test indexing a sample repo
- [ ] Smoke test API and MCP servers in the target environment

## References
- Usage, API, and CLI examples: [README](README.md)
- Configuration defaults: `codehawk/config.py`
- Release checklist: [RELEASE.md](RELEASE.md)
