using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using LocalReranker;

// Run benchmarks
BenchmarkRunner.Run<RerankerBenchmarks>();

/// <summary>
/// Performance benchmarks for LocalReranker.
/// Run with: dotnet run -c Release
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class RerankerBenchmarks
{
    private Reranker _reranker = null!;
    private string _query = null!;
    private string[] _documents10 = null!;
    private string[] _documents100 = null!;
    private string[] _documents1000 = null!;

    [GlobalSetup]
    public void Setup()
    {
        Console.WriteLine("Setting up benchmarks...");

        // Initialize reranker (will download model if needed)
        _reranker = new Reranker(new RerankerOptions
        {
            ModelId = "default",
            BatchSize = 32
        });

        // Warmup to ensure model is loaded
        _reranker.WarmupAsync().Wait();

        // Setup test data
        _query = "What are the key concepts in machine learning?";

        _documents10 = GenerateDocuments(10);
        _documents100 = GenerateDocuments(100);
        _documents1000 = GenerateDocuments(1000);

        Console.WriteLine("Setup complete!");
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _reranker?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public async Task<float[]> Score_10Documents()
    {
        return await _reranker.ScoreAsync(_query, _documents10);
    }

    [Benchmark]
    public async Task<float[]> Score_100Documents()
    {
        return await _reranker.ScoreAsync(_query, _documents100);
    }

    [Benchmark]
    public async Task<float[]> Score_1000Documents()
    {
        return await _reranker.ScoreAsync(_query, _documents1000);
    }

    [Benchmark]
    public async Task<IReadOnlyList<RankedResult>> Rerank_10Documents_Top5()
    {
        return await _reranker.RerankAsync(_query, _documents10, topK: 5);
    }

    [Benchmark]
    public async Task<IReadOnlyList<RankedResult>> Rerank_100Documents_Top10()
    {
        return await _reranker.RerankAsync(_query, _documents100, topK: 10);
    }

    [Benchmark]
    public async Task<IReadOnlyList<RankedResult>> Rerank_1000Documents_Top10()
    {
        return await _reranker.RerankAsync(_query, _documents1000, topK: 10);
    }

    private static string[] GenerateDocuments(int count)
    {
        var topics = new[]
        {
            "Machine learning is a branch of artificial intelligence that enables systems to learn from data and improve over time without explicit programming.",
            "Deep learning uses neural networks with multiple layers to extract increasingly complex features from raw input data.",
            "Natural language processing helps computers understand, interpret, and generate human language in meaningful ways.",
            "Computer vision enables machines to derive meaningful information from digital images, videos, and other visual inputs.",
            "Reinforcement learning trains agents to make sequential decisions by rewarding desired behaviors and punishing undesired ones.",
            "Supervised learning involves training models on labeled data where the desired output is known.",
            "Unsupervised learning finds hidden patterns in data without pre-existing labels.",
            "Transfer learning applies knowledge gained from one task to a different but related task.",
            "Python is the most popular programming language for machine learning due to its simplicity and extensive libraries.",
            "TensorFlow and PyTorch are the leading frameworks for building and deploying machine learning models.",
            "The weather forecast predicts sunny skies and mild temperatures for the weekend.",
            "Stock market indices showed mixed results today with technology sectors leading gains.",
            "The latest smartphone release features improved camera capabilities and longer battery life.",
            "Climate scientists warn of accelerating ice sheet melting in polar regions.",
            "New archaeological discoveries shed light on ancient civilizations in South America.",
        };

        var documents = new string[count];
        for (var i = 0; i < count; i++)
        {
            documents[i] = topics[i % topics.Length];
        }

        return documents;
    }
}

/// <summary>
/// Batch size comparison benchmarks.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class BatchSizeBenchmarks
{
    private string _query = null!;
    private string[] _documents = null!;

    [Params(8, 16, 32, 64)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _query = "What are the key concepts in machine learning?";
        _documents = Enumerable.Range(0, 100)
            .Select(i => $"This is document number {i} about various machine learning topics and applications.")
            .ToArray();
    }

    [Benchmark]
    public async Task<float[]> Score_WithBatchSize()
    {
        using var reranker = new Reranker(new RerankerOptions
        {
            ModelId = "default",
            BatchSize = BatchSize
        });

        return await reranker.ScoreAsync(_query, _documents);
    }
}
