# LocalReranker

[![NuGet](https://img.shields.io/nuget/v/LocalReranker.svg)](https://www.nuget.org/packages/LocalReranker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, zero-configuration semantic reranker for .NET. No API keys, no cloud dependencies—just install and use.

## Features

- **Zero Configuration** - Works out of the box with sensible defaults
- **Auto Model Download** - Models are downloaded and cached automatically from HuggingFace Hub
- **Offline First** - Fully local inference, no internet required after initial download
- **Minimal Dependencies** - Only 2 packages: ONNX Runtime + ML.Tokenizers
- **Cross-Platform** - Windows, Linux, macOS
- **Thread-Safe** - Single instance can be shared across multiple threads

## Installation

```bash
dotnet add package LocalReranker
```

## Quick Start

```csharp
using LocalReranker;

await using var reranker = new Reranker();

var query = "What is machine learning?";
var documents = new[]
{
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny.",
    "Deep learning uses neural networks with many layers."
};

var results = await reranker.RerankAsync(query, documents, topK: 5);

foreach (var result in results)
{
    Console.WriteLine($"[{result.Score:F4}] {result.Document}");
}
```

## Configuration

```csharp
var reranker = new Reranker(new RerankerOptions
{
    ModelId = "quality",              // Use alias or full HuggingFace ID
    MaxSequenceLength = 512,
    CacheDirectory = "/path/to/models",
    UseGpu = true,
    BatchSize = 64
});
```

## Supported Models (v0.1.0)

| Alias | Model | Size | Use Case |
|-------|-------|------|----------|
| `default` | ms-marco-MiniLM-L-6-v2 | 90MB | ✅ Balanced speed/quality (recommended) |
| `quality` | ms-marco-MiniLM-L-12-v2 | 134MB | ✅ Higher accuracy |
| `fast` | ms-marco-TinyBERT-L-2-v2 | 17MB | ✅ Fastest inference |

### Coming in v0.2.0

| Alias | Model | Size | Use Case |
|-------|-------|------|----------|
| `bge-base` | BAAI/bge-reranker-base | 440MB | 🔜 Multilingual support |
| `multilingual` | BAAI/bge-reranker-v2-m3 | 1.1GB | 🔜 Best quality, 8K context |

> **Note**: BGE models require SentencePiece tokenizer support, which is planned for v0.2.0.

## ASP.NET Core Integration

```csharp
// Program.cs - Register as singleton (recommended)
builder.Services.AddSingleton<IReranker>(sp =>
{
    var reranker = new Reranker(new RerankerOptions
    {
        ModelId = "default",
        BatchSize = 32
    });
    reranker.WarmupAsync().Wait(); // Pre-load model at startup
    return reranker;
});
```

```csharp
// SearchService.cs
public class SearchService(IReranker reranker)
{
    public async Task<List<Document>> SearchAsync(string query, List<Document> candidates)
    {
        var texts = candidates.Select(d => d.Content).ToArray();
        var ranked = await reranker.RerankAsync(query, texts, topK: 10);
        return ranked.Select(r => candidates[r.OriginalIndex]).ToList();
    }
}
```

## Performance Tips

1. **Reuse instances** - Create one `Reranker` and reuse it (thread-safe)
2. **Use WarmupAsync()** - Pre-load model during app startup
3. **Adjust BatchSize** - Larger batches are faster but use more memory
4. **Use topK** - Return only the results you need

## Requirements

- .NET 10.0+
- ~90MB disk space (default model)
- ~200MB RAM (model loaded)

## Documentation

- [API Reference](docs/API.md)
- [Design Overview](docs/DESIGN.md)
- [Changelog](CHANGELOG.md)

## License

MIT - See [LICENSE](LICENSE) for details.