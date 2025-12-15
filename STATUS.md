# Release readiness summary

This document is the authoritative release checklist for CodeHawk. It complements the installation and usage guidance in the README and replaces the previous IMPLEMENTATION notes.

## Current capabilities
- Tree-sitter–backed parsing for Python, JavaScript/TypeScript, Java, Go, and Rust, with graceful warnings when grammars are missing.
- Structure-aware chunking that prefers semantic nodes (functions, classes, methods) and falls back to overlapped line windows when a parse tree is unavailable.
- Embedding generation through `sentence-transformers` with deterministic offline fallbacks and a pluggable backend hook for custom models.
- PostgreSQL storage (with pgvector) that tracks repositories, files, chunks, relations, and lineage, plus search and context-pack assembly for LLM callers.
- Incremental-friendly indexing helpers that skip unchanged files via content hashing and remove deleted paths before writing new chunks and relations.

## Resolved defects
- Offline embedding fallback now produces deterministic vectors for tests while surfacing network errors clearly when model downloads fail.
- Indexing now removes deleted files before inserting new chunks, keeping the graph and lineage tables consistent across incremental runs.

## Open limitations and blockers
- Tree-sitter grammars must be installed separately; missing grammars disable semantic chunking for that language.
- Embedding downloads depend on external network access; offline mode is suitable only for testing.
- Search quality is limited to vector/BM25 blending without rerankers or snippet deduplication.
- API, MCP, and CLI commands assume a running PostgreSQL with pgvector; there is no bundled container or migration tool, and authentication/authorization are out of scope.
- Automated tests provide minimal coverage and do not exercise database writes, API surfaces, or long-running indexing flows.

## Release readiness checklist
- [ ] PostgreSQL + pgvector provisioned and reachable using the connection string configured in environment variables (see README for `CODEHAWK_DB_*` settings).
- [ ] Required tree-sitter grammars installed for each language you plan to index.
- [ ] Embedding model available (or offline fallback explicitly accepted) and sized to `CODEHAWK_EMBEDDING_DIMENSION`.
- [ ] Database schema initialized via `codehawk init-db` and smoke-tested by indexing a small repository through the CLI.
- [ ] API, MCP, and CLI flows validated end-to-end against your environment (follow the usage examples in the README), with observability and retries in place for long indexing jobs.
- [ ] Minimal integration verification covering indexing, search, and context-pack generation, either through automated tests or targeted manual checks.

## Authoritative references
- Primary usage guidance: [README](README.md) for installation, configuration, and CLI/API/MCP examples.
- Defaults and detailed behavior: in-code settings in `codehawk/config.py` and component modules.
