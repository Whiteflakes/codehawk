# Context Engine Status and Next Steps

## Current strengths
- **Multi-language parsing is wired up**: Tree-sitter parsers are initialized for Python, JavaScript/TypeScript, Java, Go, and Rust, aligning extensions to grammar modules so non-Python/JS code is handled semantically instead of line-by-line fallbacks.【F:codehawk/parser/__init__.py†L15-L52】
- **Semantic chunking for all advertised languages**: Function/class-aware chunking detects meaningful nodes across Python, JS/TS, Java, Go, and Rust before falling back to overlapped line blocks, keeping context slices aligned to logical units.【F:codehawk/chunker/__init__.py†L41-L178】
- **Relation- and lineage-ready context packs**: Context retrieval blends vector search with optional lexical weighting and passes retrieved chunk IDs to relation and lineage lookups so MCP/LLM callers can stitch dependency-aware packs.【F:codehawk/context/__init__.py†L204-L295】【F:codehawk/database/__init__.py†L244-L360】

## Gaps and recommended next steps
1. **Accelerate and harden indexing for large/active repos**
   - Today each file is re-read, parsed, chunked, and each chunk is embedded and inserted one-by-one with no change detection or batching. Repo walks also rescan every supported extension on each run.【F:codehawk/context/__init__.py†L121-L203】【F:codehawk/context/__init__.py†L297-L321】 
   - Add file hashing/mtime tracking to skip unchanged files, queue deletions, and stream inserts in batches (both embeddings and DB writes) to minimize round-trips and improve throughput on monorepos.

2. **Improve retrieval quality beyond basic vector/lexical blending**
   - The search path currently blends cosine similarity with optional BM25-style ranking but lacks rerankers, snippet deduplication, or structural boosts (e.g., prefer top-level definitions or recently touched chunks).【F:codehawk/database/__init__.py†L244-L360】 
   - Introduce lightweight cross-encoder reranking, repository/language priors, and duplication suppression so returned packs are higher-precision and cheaper for LLM consumption.

3. **Tighten incremental lineage and relation refresh**
   - Relation and lineage fetches assume database state is already consistent, but indexing does not reconcile moved/renamed files or recompute relations when code changes.【F:codehawk/context/__init__.py†L187-L295】 
   - Store file commit fingerprints and re-run graph analysis only for touched files/ancestors, ensuring relation graphs stay accurate without full reindexing.

4. **Reduce dependency friction for local runs**
   - Embedding generation returns `None` when the model is absent or download stalls, which can leave chunks unembedded during offline or air-gapped runs.【F:codehawk/embeddings/__init__.py†L25-L88】 
   - Ship a small default model checkpoint or allow pluggable local embedding backends (e.g., ggml/onnx) with clear error surfacing so indexing never silently drops chunks.
