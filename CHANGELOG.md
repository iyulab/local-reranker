# Changelog

All notable changes to LocalReranker will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0
- SentencePiece/Unigram tokenizer support
- BGE model family support:
  - `bge-reranker-base` - 440MB, multilingual support
  - `bge-reranker-v2-m3` - 1.1GB, best quality, 8K context

## [0.1.0] - 2025-11-26

### Added
- Initial release of LocalReranker
- Core `Reranker` class with zero-configuration support
- `IReranker` interface for dependency injection
- `RerankerOptions` for customization
- `RankedResult` struct for reranking results
- Automatic model download from HuggingFace Hub
- Local model caching with SHA256 verification
- GPU acceleration support (CUDA, DirectML, CoreML)
- Built-in model registry with preset aliases
- Batch processing for multiple documents
- Thread-safe concurrent inference

### Supported Models (v0.1.0)
- `ms-marco-MiniLM-L-6-v2` (default) - 90MB, balanced quality/speed
- `ms-marco-MiniLM-L-12-v2` (quality) - 134MB, higher accuracy
- `ms-marco-TinyBERT-L-2-v2` (fast) - 17MB, fastest inference

### Fixed
- TokenizerWrapper now correctly parses HuggingFace `tokenizer.json` format
- Vocabulary extraction from `model.vocab` section for WordPiece tokenizer

### Dependencies
- Microsoft.ML.OnnxRuntime 1.23.2
- Microsoft.ML.Tokenizers 2.0.0

### Documentation
- README.md with quick start guide
- docs/DESIGN.md with architecture overview
- docs/API.md with full API reference
- BasicUsage sample project
- AspNetCoreIntegration sample project
- BenchmarkSample for performance testing

### Design Decisions
- Minimal footprint: Only 2 NuGet dependencies
- No Microsoft.Extensions.* in core library (user handles DI registration)
- Cross-encoder architecture outputs scalar scores (not vectors)
- Lazy initialization with thread-safe warmup support
