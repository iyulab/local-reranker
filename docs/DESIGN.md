# LocalReranker Design Overview

This document provides an architectural overview of LocalReranker for users who want to understand how the library works internally.

## Core Philosophy

### Zero-Friction Adoption

```csharp
new Reranker() // Just works
```

- **No configuration required** - Sensible defaults for immediate use
- **No model selection** - Validated default model auto-selected
- **No path setup** - Platform-specific cache locations used automatically
- **No download management** - Automatic download, caching, and verification

### Minimal Footprint

```
2 NuGet packages + 1 model file = Complete solution
```

**Dependencies (2 only)**:
- `Microsoft.ML.OnnxRuntime` - Cross-platform inference engine
- `Microsoft.ML.Tokenizers` - Text tokenization

**Explicitly excluded**:
- TorchSharp (350MB+ native libraries)
- Microsoft.Extensions.* (DI is user's responsibility)
- System.Numerics.Tensors (Cross-encoder outputs scalars, not vectors)

### Offline-First

- Internet required: Only for initial model download
- Internet not required: Every subsequent run
- All inference runs locally without external API calls

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Public API Layer                         │
│  Reranker, IReranker, RerankerOptions, RankedResult        │
├─────────────────────────────────────────────────────────────┤
│                    Core Layer                               │
│  TokenizerWrapper, CrossEncoderInference, ScoreNormalizer  │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                     │
│  ModelManager, HuggingFaceClient, CacheManager             │
├─────────────────────────────────────────────────────────────┤
│                    External Dependencies                    │
│  Microsoft.ML.OnnxRuntime, Microsoft.ML.Tokenizers         │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Encoder Architecture

Unlike bi-encoders (embedding models) that encode query and document separately, cross-encoders process the query-document pair together:

```
Bi-Encoder:                    Cross-Encoder (LocalReranker):
Query  → [Encoder] → vec1      Query + Document → [Encoder] → score
Document → [Encoder] → vec2
similarity(vec1, vec2)
```

**Input format**: `[CLS] query [SEP] document [SEP]`
**Output**: Single relevance score (0.0 - 1.0)

This architecture provides higher accuracy but requires re-encoding for each query-document pair, making it ideal for reranking a small set of candidates rather than searching a large corpus.

## Key Design Decisions

### Thread Safety

- `Reranker` instances are thread-safe
- Single instance can be shared across multiple threads
- Recommended: Register as Singleton in DI containers

### Lazy Initialization

Models are loaded on first inference, not at construction:

```csharp
var reranker = new Reranker(); // Fast - no model loaded yet
await reranker.WarmupAsync();   // Optional - pre-load model
await reranker.RerankAsync(...); // Model loaded if not already
```

### GPU Fallback

GPU acceleration is best-effort with automatic CPU fallback:

```csharp
var options = new RerankerOptions { UseGpu = true };
// If GPU unavailable, silently falls back to CPU
```

### Score Normalization

Raw model outputs (logits) are normalized using sigmoid activation:

```
score = 1 / (1 + e^(-logit))
```

This produces scores in the 0.0 - 1.0 range where:
- ~1.0 = Highly relevant
- ~0.5 = Neutral
- ~0.0 = Not relevant

**Score Interpretation Guidelines**:
- Scores are **relative within a single query**, not absolute across queries
- Different models may produce different score distributions
- Recommended threshold for filtering: Start with 0.5, adjust based on your use case
- For best results, use `topK` instead of score thresholds

### Batch Processing

Documents are processed in configurable batches to balance memory usage and throughput:

```csharp
var options = new RerankerOptions
{
    BatchSize = 32  // Default: 32 documents per batch
};
```

**Batch Size Guidelines**:

| BatchSize | Memory Usage | Throughput | Recommended For |
|-----------|--------------|------------|-----------------|
| 8-16 | Low (~100MB) | Moderate | Memory-constrained environments |
| 32 | Medium (~200MB) | Good | **Default - balanced choice** |
| 64-128 | High (~400MB+) | Best | High-throughput servers with ample RAM |

**Concurrency Considerations**:
- Single `Reranker` instance is thread-safe for concurrent queries
- Each concurrent call processes its own batch independently
- For very high concurrency (>50 simultaneous queries), consider multiple instances
- Batch processing is sequential within a single `RerankAsync` call

**Memory Formula** (approximate):
```
Peak Memory ≈ Model Size + (BatchSize × MaxSequenceLength × 4 bytes × 3 tensors)
            ≈ 90MB + (32 × 512 × 4 × 3)
            ≈ 90MB + 192KB per batch ≈ ~100MB typical
```

## Supported Models

### v0.1.0 (Current)

| Alias | Model | Architecture | Tokenizer |
|-------|-------|--------------|-----------|
| `default` | ms-marco-MiniLM-L-6-v2 | BERT | WordPiece |
| `quality` | ms-marco-MiniLM-L-12-v2 | BERT | WordPiece |
| `fast` | ms-marco-TinyBERT-L-2-v2 | BERT | WordPiece |

### v0.2.0 (Planned)

| Alias | Model | Architecture | Tokenizer |
|-------|-------|--------------|-----------|
| `bge-base` | BAAI/bge-reranker-base | XLM-RoBERTa | SentencePiece |
| `multilingual` | BAAI/bge-reranker-v2-m3 | XLM-RoBERTa | SentencePiece |

## Performance Characteristics

| Metric | Target | Conditions |
|--------|--------|------------|
| Cold start | < 3s | Default model, SSD |
| Warm inference | < 50ms | 10 documents, CPU |
| Throughput | > 200 docs/sec | Batch mode, CPU |
| Memory (idle) | < 200MB | Model loaded |
| Memory (peak) | < 500MB | 100 document batch |

## Integration Patterns

### ASP.NET Core (Singleton)

```csharp
builder.Services.AddSingleton<IReranker>(sp =>
{
    var reranker = new Reranker(new RerankerOptions
    {
        ModelId = "default",
        BatchSize = 32
    });
    reranker.WarmupAsync().Wait();
    return reranker;
});
```

### RAG Pipeline

```csharp
// 1. Vector search retrieves candidates
var candidates = await vectorDb.SearchAsync(query, topK: 100);

// 2. Reranker refines results
var texts = candidates.Select(c => c.Text).ToArray();
var reranked = await reranker.RerankAsync(query, texts, topK: 10);

// 3. Return top results
return reranked.Select(r => candidates[r.OriginalIndex]).ToList();
```

## Further Reading

- [API Reference](API.md) - Complete API documentation
- [README](../README.md) - Quick start guide
- [CHANGELOG](../CHANGELOG.md) - Version history
