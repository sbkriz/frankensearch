# Changelog

All notable changes to [frankensearch](https://github.com/Dicklesworthstone/frankensearch) are documented here.

Versions listed below correspond to actual [GitHub Releases](https://github.com/Dicklesworthstone/frankensearch/releases) unless noted otherwise. Each entry links to representative commits and the release tag.

---

## [Unreleased](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.3...HEAD)

> Commits since v1.1.3 (2026-02-23) through HEAD (2026-03-21).

### Cloud API Embedding Providers

- Add pluggable cloud API embedding abstraction supporting OpenAI and Gemini backends, with HTTP transport, automatic retry, token-bucket rate limiting, and L2 normalization ([e5b7bab](https://github.com/Dicklesworthstone/frankensearch/commit/e5b7bab7d6a303c3503b6b7e99509d94b484c812), [d37d506](https://github.com/Dicklesworthstone/frankensearch/commit/d37d506da533fa7ded49ceb0b36af3e5214e0a40))
- Support query-param authentication for Gemini via `request_url()` trait method ([9846bb0](https://github.com/Dicklesworthstone/frankensearch/commit/9846bb03d4e1097138960fd500b44b83e4da426e))

### In-Memory Vector Index

- Add fully-resident in-memory vector index with f16 quantization, enabling use cases that skip disk entirely ([eee7b73](https://github.com/Dicklesworthstone/frankensearch/commit/eee7b73dee846d4601cee9b8b67ad66147e4f880))
- Add synchronous two-tier search API alongside in-memory index improvements ([e081bc5](https://github.com/Dicklesworthstone/frankensearch/commit/e081bc578ada4ed9e9dab06476b3737e3dad6541))

### WAL-Based Incremental Mutations (fsfs CLI)

- Add `append-batch`, `delete`, `compact`, and `daemon` commands for WAL-based incremental index mutation without full rebuilds ([0fadc4d](https://github.com/Dicklesworthstone/frankensearch/commit/0fadc4d3fdfc7c5f0ccafcbf5eb3f93747e375af))
- Document WAL commands in help text and shell completions ([1b88d5e](https://github.com/Dicklesworthstone/frankensearch/commit/1b88d5e46e494116defb13dde03499f22397c2bc))

### Bug Fixes

- Rename `LockError::PolledAfterCompletion` to `Cancelled` across all crates for clarity ([5151d47](https://github.com/Dicklesworthstone/frankensearch/commit/5151d473b1ff09836c89e381ed49f317f4ec7ffb))
- Handle `PolledAfterCompletion` LockError variant in embed, lexical, and rerank crates ([1359551](https://github.com/Dicklesworthstone/frankensearch/commit/1359551eaaa8c098474083738e68f8593456921c))
- Decouple release CI builds from quality gate so tag pushes produce artifacts ([9b9accb](https://github.com/Dicklesworthstone/frankensearch/commit/9b9accb51ea9c2c264aec428df886c91f71fc471))
- Exclude `tools/optimize_params` from default build; add `--recurse-submodules` to install.sh ([35becff](https://github.com/Dicklesworthstone/frankensearch/commit/35becff5ceda7865e49f3c4923410a6bcb1c61e3))
- Remove invalid `const fn` qualifiers from phase gate builder methods ([900898b](https://github.com/Dicklesworthstone/frankensearch/commit/900898b8c770df2b758793331870e82c8fd607c9))
- Fix rate limiter token drain and separate OpenAI/Gemini auto-detect paths ([e57f2de](https://github.com/Dicklesworthstone/frankensearch/commit/e57f2de8c4fe0f5e557a58be8e5e399acb03834e))

### Maintenance

- Update `asupersync` to 0.2.8 ([ba3ab85](https://github.com/Dicklesworthstone/frankensearch/commit/ba3ab85f0c8a0d713706d15556101cf32fce9683))
- Add comprehensive test coverage for fsfs update verification path ([da7acf5](https://github.com/Dicklesworthstone/frankensearch/commit/da7acf55a633b5a28e6f82118bd14b4b43ce1383))

---

## [v1.1.3](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.3) -- 2026-02-23

> **Bug fixes, PDF extraction, lite builds** -- [Full diff from v1.1.2](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.2...v1.1.3)

### New Features

- **Native PDF text extraction** -- `fsfs index` and `fsfs search` can now process PDF files directly without external tooling ([efc26cc](https://github.com/Dicklesworthstone/frankensearch/commit/efc26cc2fd5b6444efb1efcfa45cabb8a030982e))
- **Embedded-models feature flag** (`embedded-models`) for lite/offline builds that bundle models at compile time ([9fbdbd6](https://github.com/Dicklesworthstone/frankensearch/commit/9fbdbd6a82c4a0443aef5756c5b433d24cdd42d7))
- **Rank movement explanations** -- `TwoTierSearcher` now surfaces why a result moved between Phase 1 and Phase 2 ([117f955](https://github.com/Dicklesworthstone/frankensearch/commit/117f95539b4387c1a1e96516618cfc386d663b59))
- **Beautiful download progress** in the installer with file-size display ([2886323](https://github.com/Dicklesworthstone/frankensearch/commit/2886323fa2de42738e3e51681ed2f1f8fef7f53a))

### Bug Fixes

- Fix seven code-review bugs: ANSI box border rendering, `.max` vs `.min` confusion, `pdf_extract` panic guard, model ID lookup, else-if cleanup, `--expand`+`--daemon` warning ([f21ba72](https://github.com/Dicklesworthstone/frankensearch/commit/f21ba7225e0c181c0eafd1e161236a8eb4802507))
- Fix six security and correctness issues in installer and update logic ([97ac428](https://github.com/Dicklesworthstone/frankensearch/commit/97ac428f729eb830e06f19f7f6b04f2049f9506f))
- Harden release asset URL and checksum handling ([20aa045](https://github.com/Dicklesworthstone/frankensearch/commit/20aa045be191d4ce76b7dbf490362c7ae906e9b3))
- Improve searcher module reliability ([97e340d](https://github.com/Dicklesworthstone/frankensearch/commit/97e340da0ddff51ba2bdb2a76b8bab018f6b151a))

---

## [v1.1.2](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.2) -- 2026-02-22

> **Fix version reporting, update mechanism, and TUI visibility** -- [Full diff from v1.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.0...v1.1.2)

### Bug Fixes

- **Version reporting** -- binary now correctly reports its actual version instead of `v0.1.0`
- **Update mechanism** -- `fsfs update` constructs correct download URLs matching release asset naming (`fsfs-{version}-{triple}.{ext}`)
- **SHA256SUMS** -- update verification downloads the release-level checksum file instead of per-artifact sidecars
- **TUI visibility** -- running `fsfs` with no args now prints diagnostic messages when the TUI exits
- **Windows target triple** -- correct detection for `x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`

### Performance & Correctness

- Overhaul release asset system with proper Windows support and SHA256SUMS, refactor pipeline dedup logic ([7f4f8c7](https://github.com/Dicklesworthstone/frankensearch/commit/7f4f8c732f0918adcf3a3ce974fa84bf78dbae6b))
- Use `HashSet` for O(1) duplicate `doc_id` detection in `TwoTierIndexBuilder` ([8cd272a](https://github.com/Dicklesworthstone/frankensearch/commit/8cd272a9b714dfdf7ce5ad294d1095c3e24d9ce4))
- Use `BTreeSet` for deterministic path ordering in `PendingEvents` ([378af78](https://github.com/Dicklesworthstone/frankensearch/commit/378af788b60319a1715cc56a9c3e069723a258e7))
- Populate actual index path in WAL `IndexCorrupted` error for better diagnostics ([b72bfd4](https://github.com/Dicklesworthstone/frankensearch/commit/b72bfd41e4b9088dedbc382f1ebb7252e797c54b))

---

## [v1.1.1](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.1) -- 2026-02-22

> [Full diff from v1.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.1.0...v1.1.1)

**Note:** v1.1.0 and v1.1.1 point to the same commit (`82c18ff`). v1.1.1 was cut as a quick-follow release to address first-run issues discovered immediately after v1.1.0.

### Bug Fixes

- **Fix first-run hang on macOS** -- reduced filesystem probe budget (depth 2, 10K entries max) and excluded macOS system directories (`Library`, `Pictures`, `Movies`, etc.)
- **Add progress indicator** -- prints "Scanning for indexable directories..." before filesystem probe begins
- **SHA256SUMS filename** -- checksum file now has the correct name

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.1.1-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 (Apple Silicon) | `fsfs-1.1.1-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.1.1-x86_64-pc-windows-msvc.zip` |

---

## [v1.1.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.1.0) -- 2026-02-22

> **Crates.io publishing, resilient indexing, macOS support** -- [Full diff from v1.0.0](https://github.com/Dicklesworthstone/frankensearch/compare/v1.0.0...v1.1.0)

### Crates.io Publishing & Dependency Migration

- Switch all workspace dependencies from local path references to crates.io registry versions ([1fceef4](https://github.com/Dicklesworthstone/frankensearch/commit/1fceef4a5e62a33ce2b50ab5edede0516763a145))
- Bump core to v0.1.2 and index to v0.1.2 for crates.io republish ([c1676fb](https://github.com/Dicklesworthstone/frankensearch/commit/c1676fbe5b0d78f0ec0ac2735b02d2572fa8d25f))
- Add README.md files for all 12 crates ([790cbb0](https://github.com/Dicklesworthstone/frankensearch/commit/790cbb0d45a6b7cf241d44415e1e6b971f6d8700))

### Resilient Indexing Pipeline

- Checkpoint resume, embedding retries, degraded-mode completion, watcher auto-restart, and heap/normalization fixes ([4547323](https://github.com/Dicklesworthstone/frankensearch/commit/45473238fa4cdcf5b3408a5303b8efd026c48563))
- Prevent infinite loop in snapshot walker on symlink cycles ([af02a5d](https://github.com/Dicklesworthstone/frankensearch/commit/af02a5d73693b99ad59cb761c6519e3b096c319e))
- Recognize Johnson-Lindenstrauss embedders as hash embedders in storage ([1738a6b](https://github.com/Dicklesworthstone/frankensearch/commit/1738a6bd2f87da22f55eaf87acbbbd0a787911b0))

### Search Pipeline Hardening

- Harden search pipeline: fix nested markdown links, optimize diff, stabilize MMR, add bounds checks ([ee88129](https://github.com/Dicklesworthstone/frankensearch/commit/ee881297165bb75912a9a0df158d1b5d22474539))
- Improve identifier detection, WAL-first lookups, score normalization, and job queue dedup ([a74d3f2](https://github.com/Dicklesworthstone/frankensearch/commit/a74d3f2c0fe6e597cd5e7a73d191776c3bad042b))
- Model manifest expansion, index reconciliation, and storage pipeline hardening ([adc3f45](https://github.com/Dicklesworthstone/frankensearch/commit/adc3f454a189179eb95ab39bc568c761c5b9f3fd))
- Deduplicate WAL entries and fix ULID generation with telemetry prefix ([595aa48](https://github.com/Dicklesworthstone/frankensearch/commit/595aa48937d56fb48677daaa54f0f1f39dfc9d59))

### Platform Support

- **macOS arm64 (Apple Silicon)** added as a first-class release target
- Tighten root probe limits and expand excluded directories for better first-run UX ([3cdae25](https://github.com/Dicklesworthstone/frankensearch/commit/3cdae25fa2e47360edab7bed9e9e6cd3fe07db1d))

---

## [v1.0.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v1.0.0) -- 2026-02-21

> **First stable release** -- [Full diff from v0.1.0](https://github.com/Dicklesworthstone/frankensearch/compare/v0.1.0...v1.0.0)

### Highlights

First stable release of the `fsfs` CLI, marking the search engine production-ready. Two-tier hybrid local search combining lexical (Tantivy BM25) and semantic (vector cosine similarity) retrieval via Reciprocal Rank Fusion.

### Canonicalization & Reranking Overhaul

- Rewrite the query canonicalization pipeline and add fastembed-based cross-encoder reranker ([1445c78](https://github.com/Dicklesworthstone/frankensearch/commit/1445c7805cad0b6768b724a5e93b74f19cf9de65))
- Add multi-model ONNX embedder support via `OnnxEmbedderConfig` ([41c69be](https://github.com/Dicklesworthstone/frankensearch/commit/41c69be0791bf5689fee97134983a7e69e9ebdfc))

### Stability & Correctness

- Work around VDBE `sqlite_master` parameterized query limitation ([567d816](https://github.com/Dicklesworthstone/frankensearch/commit/567d8160efe1d6ef74c7070667888b174fa6e65a))
- Fix daemon modules, exclude pattern suffix matching, and clippy warnings ([2ca9040](https://github.com/Dicklesworthstone/frankensearch/commit/2ca90406e8fcf829b9a8f3ac29063916cff9fa7d))
- Resolve workspace-level dead code errors and fix 2 failing fsfs tests ([d7f144f](https://github.com/Dicklesworthstone/frankensearch/commit/d7f144f71573e04f6ae6b4173f2b787422131223))
- Resolve compilation errors, lifetime annotations, and clippy warnings across workspace ([3a280c6](https://github.com/Dicklesworthstone/frankensearch/commit/3a280c663f59e3dcaf0cbee62ea0157195467ef4))
- Fix Windows build portability and installer checksum fallback ([7f8649e](https://github.com/Dicklesworthstone/frankensearch/commit/7f8649e04a0a1953a21d9a43970d12611f3a681c))

### Platforms

| Platform | Asset |
|----------|-------|
| Linux x86_64 | `fsfs-1.0.0-x86_64-unknown-linux-musl.tar.xz` |
| macOS arm64 | `fsfs-1.0.0-aarch64-apple-darwin.tar.xz` |
| Windows x86_64 | `fsfs-1.0.0-x86_64-pc-windows-msvc.zip` |

---

## [v0.1.0](https://github.com/Dicklesworthstone/frankensearch/releases/tag/v0.1.0) -- 2026-02-19

> **Initial public release** -- [Full diff from initial commit](https://github.com/Dicklesworthstone/frankensearch/compare/42156796c494...v0.1.0)

### What Shipped

The first public release of frankensearch -- a two-tier hybrid local search engine for Rust with the `fsfs` standalone CLI. This release includes the full workspace of 11 crates and a massive amount of foundational work spanning ~200 commits over 6 days of intensive multi-agent development.

### Architecture (11-Crate Workspace)

| Crate | Responsibility |
|-------|---------------|
| `frankensearch` | Facade crate with top-level public API and re-exports |
| `frankensearch-core` | Shared types, traits, errors, config, query canonicalization/classification, metrics/eval helpers |
| `frankensearch-embed` | Embedding backends and fallback stack (`hash`, `model2vec`, `fastembed`), streaming model downloads, manifest validation |
| `frankensearch-index` | FSVI vector storage, SIMD dot products, top-k search, WAL-aware soft delete, optional ANN (HNSW) support |
| `frankensearch-lexical` | Tantivy schema/index/search for BM25 lexical retrieval |
| `frankensearch-fusion` | RRF fusion, two-tier orchestration, blending, adaptive fusion, PRF, MMR, circuit-breaker, federated search |
| `frankensearch-rerank` | Cross-encoder reranking integration |
| `frankensearch-storage` | FrankenSQLite metadata persistence, dedup/content-hash tracking, embedding queue |
| `frankensearch-durability` | Repair/protection primitives for index artifacts and segment health |
| `frankensearch-tui` | Shared TUI shell, input, theme, replay framework |
| `frankensearch-ops` | Fleet observability/control-plane TUI and telemetry materialization |

### Search Engine Capabilities

- **Two-tier progressive search** -- fast initial results (<15 ms target) followed by quality-refined results (~150 ms target)
- **Reciprocal Rank Fusion (RRF)** combining lexical BM25 and semantic vector similarity
- **Score blending** with configurable quality weight between fast and quality tiers
- **Graceful degradation** -- `RefinementFailed` phase when quality tier errors/times out, preserving Phase 1 results
- **Deterministic ordering** for reproducible ranking via stable tie-break logic
- **Query classification** (`identifier`, `short keyword`, `natural language`) with adaptive budgets
- **Exclusion queries** (`-term`) for filtering unwanted results
- **Negation-aware canonicalization** pipeline

### FSFS CLI Product

- `fsfs search` with progressive delivery (`--stream`, `--format jsonl/toon/csv/table/json`)
- `fsfs index` with filesystem watching and incremental updates
- `fsfs explain` for result explanation surfaces
- `fsfs doctor` for health checks
- `fsfs status` for runtime diagnostics
- Daemon transport with query caching, unbounded-recall tuning, and fallback when daemon unavailable ([e4870f8](https://github.com/Dicklesworthstone/frankensearch/commit/e4870f82809ea42969ddf05a513ee9a60ab35321))
- Auto-detect output format for non-TTY environments
- Adaptive debounce engine for search execution

### Vector Index (FSVI Format)

- Memory-mapped on-disk format with f16 quantization by default
- SIMD dot products with NaN-safe total ordering
- Heap-based top-k selection
- WAL journaling for crash recovery
- HNSW approximate nearest-neighbor path (optional, for larger corpora)
- Soft-delete with rollback support

### Embedding System

- Hash embedder (zero model downloads) for dev/CI
- `model2vec` embedder (potion-multilingual-128M for fast tier)
- `fastembed` embedder (all-MiniLM-L6-v2 for quality tier)
- Automatic embedder stack detection and fallback
- Streaming model downloads with manifest validation and batch size clamping
- Filesystem-backed verification cache for model manifests
- Dimension reduction support

### Storage & Durability

- FrankenSQLite-backed metadata persistence with content-hash dedup
- Immediate transactions, ingest pipeline, staleness detection
- Concurrent schema bootstrap with race-safe upsert
- File protector with e2e corruption recovery and atomic operations
- fsync hardening for all production metadata writes

### Fusion Pipeline

- Adaptive fusion with circuit breaker and phase gating
- Pseudo-relevance feedback (PRF) and Maximal Marginal Relevance (MMR)
- Conformal calibration for score confidence
- Graph-augmented query coordination with query-biased PageRank
- NaN/Infinity fallback in RRF scoring

### Ops Dashboard & Telemetry

- Multi-screen TUI dashboard (alerts/SLO, fleet, resources, analytics, timeline)
- Telemetry ingest pipeline with backpressure and attribution
- Control-plane health alerting and self-monitoring
- Lifecycle tracking and discovery with PID refresh
- FrankenSQLite-backed telemetry storage

### Cross-Platform

- Linux x86_64 (`musl` static binary)
- Windows x86_64 (zip + standalone exe)
- curl|bash installer with HTTP proxy forwarding
- Background daemon service installation (systemd / launchd / schtasks) ([83f3298](https://github.com/Dicklesworthstone/frankensearch/commit/83f32989373b16f3c950dd6220568cad7be4a185))
- Windows CI build target and zip archive packaging ([af41cdc](https://github.com/Dicklesworthstone/frankensearch/commit/af41cdc3))

### Build System & Quality

- Feature-flag tier system: `default` (hash), `semantic`, `hybrid`, `persistent`, `durable`, `full`, `full-fts5`
- CMA-ES hyperparameter optimizer for fusion pipeline tuning
- IR evaluation metrics: nDCG@K, MRR, Recall@K, MAP with bootstrap confidence intervals
- Benchmark baseline matrix and pressure simulation harness
- MIT License with OpenAI/Anthropic Rider ([9f60b71](https://github.com/Dicklesworthstone/frankensearch/commit/9f60b71d))
- Published to crates.io (core crates v0.1.1) ([a8d64a0](https://github.com/Dicklesworthstone/frankensearch/commit/a8d64a06))

### Concurrency Model

- Built on `asupersync` and capability context (`Cx`), not Tokio
- Cancellation-aware search phases and timeouts
- Graceful lock recovery (no panic-on-poison) across all concurrent components ([968a6b8](https://github.com/Dicklesworthstone/frankensearch/commit/968a6b8f))
- Zero-alloc byte-level SIMD dot products eliminating thread-local scratch buffers ([3ae2caa](https://github.com/Dicklesworthstone/frankensearch/commit/3ae2caa7))

---

## Pre-Release History

> 2026-02-13 to 2026-02-18. Project scaffolding and planning phase before the first tagged release.

- Initial commit: project scaffolding, README, AGENTS.md, and beads task graph ([4215679](https://github.com/Dicklesworthstone/frankensearch/commit/42156796c494a74d4dfad83301d25bec04058c61))
- Document `asupersync` as mandatory async runtime, purge Tokio references ([3f29794](https://github.com/Dicklesworthstone/frankensearch/commit/3f297945))
- Initialize Rust workspace with six-crate hybrid search architecture ([f39c793](https://github.com/Dicklesworthstone/frankensearch/commit/f39c793a))
- Implement full search pipeline across embed, index, lexical, fusion, and rerank crates ([3965991](https://github.com/Dicklesworthstone/frankensearch/commit/39659917))
- Add storage, durability, fsfs, tui, and ops crates to workspace ([484f9cc](https://github.com/Dicklesworthstone/frankensearch/commit/484f9cc9))
