# LocalReranker API Reference

## Table of Contents

- [Classes](#classes)
  - [Reranker](#reranker)
  - [RerankerOptions](#rerankeroptions)
  - [RankedResult](#rankedresult)
- [Interfaces](#interfaces)
  - [IReranker](#ireranker)
- [Enums](#enums)
  - [GpuProvider](#gpuprovider)
  - [OutputShape](#outputshape)
  - [ModelArchitecture](#modelarchitecture)
- [Models](#models)
  - [ModelInfo](#modelinfo)
  - [ModelRegistry](#modelregistry)
- [Exceptions](#exceptions)

---

## Classes

### Reranker

The main class for semantic document reranking.

```csharp
public sealed class Reranker : IReranker
```

#### Constructors

| Constructor | Description |
|-------------|-------------|
| `Reranker()` | Initializes with default settings (ms-marco-MiniLM-L-6-v2 model) |
| `Reranker(RerankerOptions options)` | Initializes with custom configuration |

#### Methods

| Method | Description |
|--------|-------------|
| `RerankAsync(string query, IEnumerable<string> documents, int? topK, CancellationToken)` | Reranks documents by relevance |
| `ScoreAsync(string query, IEnumerable<string> documents, CancellationToken)` | Returns relevance scores without sorting |
| `RerankBatchAsync(IEnumerable<string> queries, IEnumerable<IEnumerable<string>> documentSets, int? topK, CancellationToken)` | Batch reranking for multiple queries |
| `WarmupAsync(CancellationToken)` | Pre-loads model to avoid cold start |
| `GetModelInfo()` | Returns loaded model information |
| `Dispose()` | Releases resources |
| `DisposeAsync()` | Asynchronously releases resources |

#### Example

```csharp
using LocalReranker;

// Zero-configuration usage
await using var reranker = new Reranker();

var results = await reranker.RerankAsync(
    "What is machine learning?",
    documents,
    topK: 5);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

---

### RerankerOptions

Configuration options for the Reranker.

```csharp
public sealed class RerankerOptions
```

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `ModelId` | `string` | `"default"` | Model identifier (alias, HuggingFace ID, or path) |
| `MaxSequenceLength` | `int?` | `null` | Maximum input length (null uses model default) |
| `CacheDirectory` | `string?` | `null` | Custom model cache directory |
| `UseGpu` | `bool` | `false` | Enable GPU acceleration |
| `GpuProvider` | `GpuProvider` | `Auto` | Preferred GPU provider |
| `DisableAutoDownload` | `bool` | `false` | Disable automatic model download |
| `ThreadCount` | `int?` | `null` | Inference threads (null uses ProcessorCount) |
| `BatchSize` | `int` | `32` | Batch size for processing |

#### Model Aliases (v0.1.0)

| Alias | Model | Size | Description |
|-------|-------|------|-------------|
| `"default"` | ms-marco-MiniLM-L-6-v2 | 90MB | Balanced quality/speed (recommended) |
| `"quality"` | ms-marco-MiniLM-L-12-v2 | 134MB | Higher accuracy |
| `"fast"` | ms-marco-TinyBERT-L-2-v2 | 17MB | Fastest inference |

#### Coming in v0.2.0

| Alias | Model | Size | Description |
|-------|-------|------|-------------|
| `"bge-base"` | BAAI/bge-reranker-base | 440MB | Multilingual support |
| `"multilingual"` | BAAI/bge-reranker-v2-m3 | 1.1GB | Best quality, 8K context |

> **Note**: BGE models require SentencePiece tokenizer support, planned for v0.2.0.

#### Example

```csharp
var options = new RerankerOptions
{
    ModelId = "quality",
    MaxSequenceLength = 256,
    UseGpu = true,
    BatchSize = 64
};

var reranker = new Reranker(options);
```

---

### RankedResult

Represents a reranked document with its relevance score.

```csharp
public readonly record struct RankedResult(
    int OriginalIndex,
    float Score,
    string Document
) : IComparable<RankedResult>
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `OriginalIndex` | `int` | Zero-based index in original input array |
| `Score` | `float` | Relevance score (0.0 to 1.0) |
| `Document` | `string` | Original document text |

#### Example

```csharp
var results = await reranker.RerankAsync(query, documents);

foreach (var result in results)
{
    // Get original document by index
    var originalDoc = myDocuments[result.OriginalIndex];

    Console.WriteLine($"Score: {result.Score:P2}");
    Console.WriteLine($"Document: {result.Document}");
}
```

---

## Interfaces

### IReranker

Interface for semantic document reranking.

```csharp
public interface IReranker : IDisposable, IAsyncDisposable
```

#### Methods

| Method | Return Type | Description |
|--------|-------------|-------------|
| `RerankAsync(query, documents, topK?, cancellationToken)` | `Task<IReadOnlyList<RankedResult>>` | Reranks documents by relevance |
| `ScoreAsync(query, documents, cancellationToken)` | `Task<float[]>` | Returns scores without sorting |
| `RerankBatchAsync(queries, documentSets, topK?, cancellationToken)` | `Task<IReadOnlyList<IReadOnlyList<RankedResult>>>` | Batch reranking |
| `WarmupAsync(cancellationToken)` | `Task` | Pre-loads model |
| `GetModelInfo()` | `ModelInfo?` | Gets loaded model info |

#### DI Registration Example

```csharp
// Register as singleton (recommended)
services.AddSingleton<IReranker>(sp => new Reranker(new RerankerOptions
{
    ModelId = "default",
    UseGpu = true
}));
```

---

## Enums

### GpuProvider

GPU provider options for hardware acceleration.

```csharp
public enum GpuProvider
{
    Auto,      // Auto-select best available
    Cuda,      // NVIDIA CUDA
    DirectML,  // DirectX 12 (Windows)
    CoreML,    // Apple Silicon
    Cpu        // Force CPU
}
```

### OutputShape

Output tensor shape types (internal use).

```csharp
public enum OutputShape
{
    SingleLogit,         // [batch_size, 1]
    BinaryClassification, // [batch_size, 2]
    FlatLogit            // [batch_size]
}
```

### ModelArchitecture

Model architecture types.

```csharp
public enum ModelArchitecture
{
    Bert,       // BERT with WordPiece
    Roberta,    // RoBERTa with BPE
    XlmRoberta, // Multilingual XLM-RoBERTa
    JinaBert    // JinaBERT with ALiBi
}
```

---

## Models

### ModelInfo

Contains metadata and configuration for a reranker model.

```csharp
public sealed record ModelInfo
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Id` | `string` | Unique identifier (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2") |
| `Alias` | `string` | Short alias (e.g., "default") |
| `DisplayName` | `string` | Human-readable name |
| `Parameters` | `long` | Number of model parameters |
| `MaxSequenceLength` | `int` | Maximum input length |
| `SizeBytes` | `long` | Model size in bytes |
| `SizeMB` | `double` | Model size in megabytes |
| `IsMultilingual` | `bool` | Multilingual support |
| `Architecture` | `ModelArchitecture` | Model architecture type |
| `OutputShape` | `OutputShape` | Expected output shape |

### ModelRegistry

Registry for looking up model information.

```csharp
public sealed class ModelRegistry
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Default` | `ModelRegistry` | Default registry with built-in models |

#### Methods

| Method | Description |
|--------|-------------|
| `Resolve(string modelIdOrAlias)` | Resolves model ID/alias to ModelInfo |
| `TryResolve(string modelIdOrAlias, out ModelInfo?)` | Tries to resolve without throwing |
| `GetAll()` | Returns all registered models |
| `GetAliases()` | Returns all available aliases |

---

## Exceptions

### LocalRerankerException

Base exception for LocalReranker errors.

```csharp
public class LocalRerankerException : Exception
```

### ModelNotFoundException

Thrown when a model cannot be found.

```csharp
public class ModelNotFoundException : LocalRerankerException
{
    public string ModelId { get; }
}
```

### InferenceException

Thrown when model inference fails.

```csharp
public class InferenceException : LocalRerankerException
```

### DownloadException

Thrown when model download fails.

```csharp
public class DownloadException : LocalRerankerException
```

---

## Thread Safety

The `Reranker` class is thread-safe. A single instance can be shared across multiple threads for concurrent inference. For best performance, create one instance and reuse it throughout your application's lifetime.

```csharp
// Thread-safe concurrent usage
var tasks = queries.Select(q => reranker.RerankAsync(q, documents));
var results = await Task.WhenAll(tasks);
```

---

## Performance Tips

1. **Reuse Reranker instances** - Avoid creating new instances per request
2. **Use WarmupAsync** - Pre-load models during application startup
3. **Adjust BatchSize** - Larger batches are faster but use more memory
4. **Enable GPU** - Set `UseGpu = true` for CUDA/DirectML acceleration
5. **Limit sequences** - Use `MaxSequenceLength` to truncate long documents
6. **Use topK** - Return only needed results instead of all documents
