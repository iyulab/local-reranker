using FluentAssertions;
using LocalReranker.Infrastructure;
using Xunit;

namespace LocalReranker.Tests;

public class CacheManagerTests : IDisposable
{
    private readonly string _testCacheDir;
    private readonly CacheManager _cacheManager;

    public CacheManagerTests()
    {
        _testCacheDir = Path.Combine(Path.GetTempPath(), $"LocalReranker_Test_{Guid.NewGuid():N}");
        _cacheManager = new CacheManager(_testCacheDir);
    }

    [Fact]
    public void CacheDirectory_ShouldUseCustomPath()
    {
        _cacheManager.CacheDirectory.Should().Be(_testCacheDir);
    }

    [Fact]
    public void GetModelDirectory_ShouldFollowHuggingFaceCacheStructure()
    {
        var dir = _cacheManager.GetModelDirectory("cross-encoder/ms-marco-MiniLM-L-6-v2");

        // HuggingFace cache structure: models--{org}--{model}/snapshots/{revision}
        dir.Should().Contain("models--cross-encoder--ms-marco-MiniLM-L-6-v2");
        dir.Should().Contain("snapshots");
        dir.Should().EndWith("main");
    }

    [Fact]
    public void GetModelDirectory_WithRevision_ShouldIncludeRevision()
    {
        var dir = _cacheManager.GetModelDirectory("some/model", "v1.0");

        // HuggingFace cache structure: models--{org}--{model}/snapshots/{revision}
        dir.Should().Contain(Path.Combine("models--some--model", "snapshots", "v1.0"));
    }

    [Fact]
    public void GetModelFilePath_ShouldReturnCorrectPath()
    {
        var path = _cacheManager.GetModelFilePath("org/model", "model.onnx");

        path.Should().EndWith("model.onnx");
        path.Should().Contain("models--org--model");
    }

    [Fact]
    public void EnsureModelDirectory_ShouldCreateDirectory()
    {
        var dir = _cacheManager.EnsureModelDirectory("test/model");

        Directory.Exists(dir).Should().BeTrue();
    }

    [Fact]
    public void ModelFileExists_NonExistent_ShouldReturnFalse()
    {
        var exists = _cacheManager.ModelFileExists("nonexistent/model", "model.onnx");

        exists.Should().BeFalse();
    }

    [Fact]
    public void ModelFileExists_Existing_ShouldReturnTrue()
    {
        var dir = _cacheManager.EnsureModelDirectory("test/model");
        var filePath = Path.Combine(dir, "model.onnx");
        File.WriteAllText(filePath, "test");

        var exists = _cacheManager.ModelFileExists("test/model", "model.onnx");

        exists.Should().BeTrue();
    }

    [Fact]
    public void DeleteModel_ShouldRemoveDirectory()
    {
        var dir = _cacheManager.EnsureModelDirectory("delete/test");
        File.WriteAllText(Path.Combine(dir, "file.txt"), "test");

        _cacheManager.DeleteModel("delete/test");

        Directory.Exists(Path.Combine(_testCacheDir, "models--delete--test")).Should().BeFalse();
    }

    [Fact]
    public void GetCachedModels_ShouldListAllModels()
    {
        _cacheManager.EnsureModelDirectory("model1/test", "main");
        _cacheManager.EnsureModelDirectory("model2/test", "v1");

        var models = _cacheManager.GetCachedModels().ToList();

        models.Should().HaveCount(2);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testCacheDir))
        {
            Directory.Delete(_testCacheDir, recursive: true);
        }
    }
}
