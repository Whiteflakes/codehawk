# Release Notes

These release notes summarize the current state of CodeHawk, highlighting working capabilities, known gaps, and the steps still required for a stable release.

## Current capabilities
- Tree-sitter–backed parsing for Python, JavaScript/TypeScript, Java, Go, and Rust with graceful warnings when grammars are missing.
- Structure-aware chunking that prefers semantic nodes (functions, classes, methods) and falls back to overlapped line windows when a parse tree is unavailable.
- Embedding generation through `sentence-transformers` with deterministic offline fallbacks and a pluggable backend hook for custom models.
- PostgreSQL storage (with pgvector) that tracks repositories, files, chunks, relations, and lineage, plus basic search and context-pack assembly for LLM callers.
- Incremental-friendly indexing helpers that skip unchanged files via content hashing and remove deleted paths before writing new chunks and relations.

## Known issues and limitations
- Tree-sitter grammars must be installed separately; missing grammars log warnings and disable semantic chunking for that language.
- Embedding downloads can fail in network-restricted environments; offline mode yields deterministic vectors suitable for testing but not production relevance.
- Search quality is limited to vector/BM25 blending without rerankers or snippet deduplication, and relation/lineage enrichment depends on a healthy database state.
- API, MCP, and CLI commands assume a running PostgreSQL with pgvector; there is no bundled container or migration tool, and authentication/authorization are out of scope.
- Test coverage is minimal and does not exercise the database, API surface, or long-running indexing flows.

## Remaining work
- Package tree-sitter grammars or add setup scripts so semantic parsing works out of the box.
- Harden embedding configuration: allow explicit local model paths, clearer error surfacing, and better retries before falling back to mock vectors.
- Improve retrieval quality with rerankers, snippet deduplication, and structural boosts; add evaluation data to guard regressions.
- Ship operational tooling (schema migrations, health checks, seed data) and validate API/MCP/CLI paths end-to-end against a real database.
- Expand automated tests to cover database writes, search results, and graph/lineage reconciliation across incremental indexes.

## Release readiness checklist
- [ ] PostgreSQL + pgvector provisioned and reachable using the connection string in `CODEHAWK_DATABASE_URL`.
- [ ] Required tree-sitter grammars installed for the languages you plan to index.
- [ ] Embedding model available (or offline fallback explicitly accepted) and sized to `CODEHAWK_EMBEDDING_DIMENSION`.
- [ ] Initial database schema created via `codehawk init-db` and smoke-tested with a small repository index.
- [ ] API/MCP/CLI flows validated against your environment, with observability and retries in place for long indexing jobs.
- [ ] Minimal integration test suite executed (or equivalent manual verification) covering indexing, search, and context-pack generation.

## Authoritative references
- See the [README](README.md) for installation, configuration, and CLI/API usage examples.
- Refer to in-code defaults in `codehawk/config.py` and component modules for exact behaviors when settings are omitted.
