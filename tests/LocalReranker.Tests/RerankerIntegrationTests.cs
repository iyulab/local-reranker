using FluentAssertions;
using LocalReranker.Exceptions;
using LocalReranker.Models;
using Xunit;

namespace LocalReranker.Tests;

/// <summary>
/// Integration tests for the Reranker.
/// Tests marked with [Trait("Category", "Integration")] require model files
/// which may not be available in CI environments.
/// </summary>
public class RerankerIntegrationTests : IDisposable
{
    private Reranker? _reranker;

    #region Validation Tests (No Model Required)

    [Fact]
    public void Constructor_WithNullOptions_ShouldThrowArgumentNullException()
    {
        // Act
        var act = () => new Reranker(null!);

        // Assert
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public async Task RerankAsync_WithNullQuery_ShouldThrowArgumentNullException()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });
        var documents = new[] { "doc1", "doc2" };

        // Act
        var act = async () => await _reranker.RerankAsync(null!, documents);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task RerankAsync_WithNullDocuments_ShouldThrowArgumentNullException()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });

        // Act
        var act = async () => await _reranker.RerankAsync("query", null!);

        // Assert
        await act.Should().ThrowAsync<ArgumentNullException>();
    }

    [Fact]
    public async Task RerankAsync_WithEmptyQuery_ShouldThrowArgumentException()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });
        var documents = new[] { "doc1", "doc2" };

        // Act
        var act = async () => await _reranker.RerankAsync("", documents);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task RerankAsync_WithWhitespaceQuery_ShouldThrowArgumentException()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });
        var documents = new[] { "doc1", "doc2" };

        // Act
        var act = async () => await _reranker.RerankAsync("   ", documents);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task RerankAsync_WithEmptyDocuments_ShouldThrowArgumentException()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });

        // Act
        var act = async () => await _reranker.RerankAsync("query", Array.Empty<string>());

        // Assert
        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task RerankBatchAsync_WithMismatchedCounts_ShouldThrowArgumentException()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });
        var queries = new[] { "q1", "q2" };
        var documentSets = new[] { new[] { "doc1" } }; // Only 1 set, but 2 queries

        // Act
        var act = async () => await _reranker.RerankBatchAsync(queries, documentSets);

        // Assert
        await act.Should().ThrowAsync<ArgumentException>()
            .WithMessage("*Number of queries*");
    }

    [Fact]
    public void GetModelInfo_BeforeInitialization_ShouldReturnNull()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });

        // Act
        var modelInfo = _reranker.GetModelInfo();

        // Assert
        modelInfo.Should().BeNull();
    }

    #endregion

    #region ModelRegistry Tests

    [Fact]
    public void ModelRegistry_Resolve_DefaultAlias_ShouldReturnDefaultModel()
    {
        // Act
        var model = ModelRegistry.Default.Resolve("default");

        // Assert
        model.Should().NotBeNull();
        model.Id.Should().Be("cross-encoder/ms-marco-MiniLM-L-6-v2");
    }

    [Fact]
    public void ModelRegistry_Resolve_QualityAlias_ShouldReturnQualityModel()
    {
        // Act
        var model = ModelRegistry.Default.Resolve("quality");

        // Assert
        model.Should().NotBeNull();
        model.Id.Should().Be("cross-encoder/ms-marco-MiniLM-L-12-v2");
    }

    [Fact]
    public void ModelRegistry_Resolve_FastAlias_ShouldReturnFastModel()
    {
        // Act
        var model = ModelRegistry.Default.Resolve("fast");

        // Assert
        model.Should().NotBeNull();
        model.Id.Should().Contain("TinyBERT");
    }

    [Fact]
    public void ModelRegistry_Resolve_MultilingualAlias_ShouldReturnMultilingualModel()
    {
        // Act
        var model = ModelRegistry.Default.Resolve("multilingual");

        // Assert
        model.Should().NotBeNull();
        model.IsMultilingual.Should().BeTrue();
    }

    [Fact]
    public void ModelRegistry_Resolve_UnknownModel_ShouldThrow()
    {
        // Act
        var act = () => ModelRegistry.Default.Resolve("unknown-model-12345");

        // Assert
        act.Should().Throw<ModelNotFoundException>()
            .Which.ModelId.Should().Be("unknown-model-12345");
    }

    [Fact]
    public void ModelRegistry_GetAll_ShouldReturnMultipleModels()
    {
        // Act
        var models = ModelRegistry.Default.GetAll();

        // Assert
        models.Should().HaveCountGreaterThanOrEqualTo(5);
    }

    #endregion

    #region Concurrency Safety Tests

    [Fact]
    public void MultipleRerankerInstances_ShouldNotInterfere()
    {
        // Arrange & Act - creating multiple instances should not throw
        var reranker1 = new Reranker(new RerankerOptions { DisableAutoDownload = true });
        var reranker2 = new Reranker(new RerankerOptions { DisableAutoDownload = true });
        var reranker3 = new Reranker(new RerankerOptions { DisableAutoDownload = true });

        // Assert
        reranker1.Should().NotBeSameAs(reranker2);
        reranker2.Should().NotBeSameAs(reranker3);

        // Cleanup
        reranker1.Dispose();
        reranker2.Dispose();
        reranker3.Dispose();
    }

    [Fact]
    public void Dispose_MultipleCalls_ShouldNotThrow()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });

        // Act & Assert - multiple dispose calls should not throw
        _reranker.Dispose();
        _reranker.Dispose();
        _reranker.Dispose();
    }

    [Fact]
    public async Task DisposeAsync_MultipleCalls_ShouldNotThrow()
    {
        // Arrange
        _reranker = new Reranker(new RerankerOptions { DisableAutoDownload = true });

        // Act & Assert - multiple dispose calls should not throw
        await _reranker.DisposeAsync();
        await _reranker.DisposeAsync();
        await _reranker.DisposeAsync();
    }

    #endregion

    #region End-to-End Integration Tests (Require Models)

    /// <summary>
    /// End-to-end reranking test with actual model.
    /// Requires model files to be available (downloaded or cached).
    /// </summary>
    [Trait("Category", "Integration")]
    [Fact(Skip = "Requires model download - run locally with models")]
    public async Task RerankAsync_WithRealModel_ShouldReturnRankedResults()
    {
        // Arrange
        _reranker = new Reranker();
        var query = "What is machine learning?";
        var documents = new[]
        {
            "Machine learning is a subset of artificial intelligence.",
            "The weather today is sunny and warm.",
            "Deep learning uses neural networks with many layers.",
            "I like pizza with cheese and tomatoes."
        };

        // Act
        var results = await _reranker.RerankAsync(query, documents);

        // Assert
        results.Should().HaveCount(4);
        results.Should().BeInDescendingOrder(r => r.Score);

        // ML-related documents should rank higher
        var topTwo = results.Take(2).Select(r => r.Document).ToList();
        topTwo.Should().Contain(d => d.Contains("Machine learning") || d.Contains("Deep learning"));
    }

    /// <summary>
    /// Tests topK limiting functionality.
    /// </summary>
    [Trait("Category", "Integration")]
    [Fact(Skip = "Requires model download - run locally with models")]
    public async Task RerankAsync_WithTopK_ShouldLimitResults()
    {
        // Arrange
        _reranker = new Reranker();
        var query = "programming languages";
        var documents = new[]
        {
            "Python is a popular programming language.",
            "Java is used for enterprise applications.",
            "JavaScript runs in browsers.",
            "The cat sat on the mat.",
            "C# is developed by Microsoft."
        };

        // Act
        var results = await _reranker.RerankAsync(query, documents, topK: 3);

        // Assert
        results.Should().HaveCount(3);
        results.Should().BeInDescendingOrder(r => r.Score);
    }

    /// <summary>
    /// Tests batch reranking with multiple queries.
    /// </summary>
    [Trait("Category", "Integration")]
    [Fact(Skip = "Requires model download - run locally with models")]
    public async Task RerankBatchAsync_WithMultipleQueries_ShouldReturnAllResults()
    {
        // Arrange
        _reranker = new Reranker();
        var queries = new[] { "capital cities", "programming" };
        var documentSets = new[]
        {
            new[] { "Paris is the capital of France", "Dogs are pets" },
            new[] { "Python is a language", "Weather is nice" }
        };

        // Act
        var results = await _reranker.RerankBatchAsync(queries, documentSets);

        // Assert
        results.Should().HaveCount(2);
        results[0].Should().HaveCount(2);
        results[1].Should().HaveCount(2);
    }

    /// <summary>
    /// Tests warmup functionality.
    /// </summary>
    [Trait("Category", "Integration")]
    [Fact(Skip = "Requires model download - run locally with models")]
    public async Task WarmupAsync_ShouldInitializeModel()
    {
        // Arrange
        _reranker = new Reranker();

        // Act
        await _reranker.WarmupAsync();
        var modelInfo = _reranker.GetModelInfo();

        // Assert
        modelInfo.Should().NotBeNull();
        modelInfo!.Id.Should().NotBeNullOrEmpty();
    }

    /// <summary>
    /// Tests concurrent access to the same Reranker instance.
    /// </summary>
    [Trait("Category", "Integration")]
    [Fact(Skip = "Requires model download - run locally with models")]
    public async Task RerankAsync_ConcurrentCalls_ShouldBeThreadSafe()
    {
        // Arrange
        _reranker = new Reranker();
        var query = "test query";
        var documents = new[] { "doc1", "doc2", "doc3" };

        // Act - make 10 concurrent calls
        var tasks = Enumerable.Range(0, 10)
            .Select(_ => _reranker.RerankAsync(query, documents))
            .ToList();

        var results = await Task.WhenAll(tasks);

        // Assert - all results should be valid
        results.Should().HaveCount(10);
        foreach (var result in results)
        {
            result.Should().HaveCount(3);
        }
    }

    /// <summary>
    /// Tests cancellation support.
    /// </summary>
    [Trait("Category", "Integration")]
    [Fact(Skip = "Requires model download - run locally with models")]
    public async Task RerankAsync_WithCancellation_ShouldThrowOperationCanceledException()
    {
        // Arrange
        _reranker = new Reranker();
        var query = "test";
        var documents = Enumerable.Range(0, 100).Select(i => $"Document {i}").ToArray();
        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Cancel immediately

        // Act
        var act = async () => await _reranker.RerankAsync(query, documents, cancellationToken: cts.Token);

        // Assert
        await act.Should().ThrowAsync<OperationCanceledException>();
    }

    #endregion

    #region GPU Fallback Tests

    [Fact]
    public void RerankerOptions_UseGpu_WithCpuProvider_ShouldFallbackToCpu()
    {
        // Arrange
        var options = new RerankerOptions
        {
            UseGpu = true,
            GpuProvider = GpuProvider.Cpu,
            DisableAutoDownload = true
        };

        // Act - creating reranker should not throw
        var reranker = new Reranker(options);

        // Assert
        reranker.Should().NotBeNull();
        reranker.Dispose();
    }

    [Fact]
    public void RerankerOptions_GpuProvider_AllValues_ShouldBeDefined()
    {
        // Assert
        GpuProvider.Auto.Should().BeDefined();
        GpuProvider.Cuda.Should().BeDefined();
        GpuProvider.DirectML.Should().BeDefined();
        GpuProvider.CoreML.Should().BeDefined();
        GpuProvider.Cpu.Should().BeDefined();
    }

    #endregion

    public void Dispose()
    {
        _reranker?.Dispose();
        GC.SuppressFinalize(this);
    }
}
