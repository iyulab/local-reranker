using LocalReranker.Exceptions;
using LocalReranker.Models;

// Use the DownloadProgress from root namespace
using DownloadProgress = LocalReranker.DownloadProgress;

namespace LocalReranker.Infrastructure;

/// <summary>
/// Manages model lifecycle including download, caching, and verification.
/// </summary>
internal sealed class ModelManager : IDisposable
{
    private readonly CacheManager _cacheManager;
    private readonly HuggingFaceClient _downloadClient;
    private readonly bool _autoDownloadEnabled;
    private bool _disposed;

    /// <summary>
    /// Initializes a new ModelManager instance.
    /// </summary>
    /// <param name="cacheDirectory">Custom cache directory, or null for default.</param>
    /// <param name="autoDownload">Whether to automatically download missing models.</param>
    public ModelManager(
        string? cacheDirectory = null,
        bool autoDownload = true)
    {
        _cacheManager = new CacheManager(cacheDirectory);
        _downloadClient = new HuggingFaceClient();
        _autoDownloadEnabled = autoDownload;
    }

    /// <summary>
    /// Ensures a model is available locally, downloading if necessary.
    /// </summary>
    /// <param name="modelInfo">Model information.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Paths to the model and tokenizer files.</returns>
    public async Task<ModelPaths> EnsureModelAsync(
        ModelInfo modelInfo,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(modelInfo);

        var modelPath = _cacheManager.GetModelFilePath(modelInfo.Id, modelInfo.OnnxFile);
        var tokenizerPath = _cacheManager.GetModelFilePath(modelInfo.Id, modelInfo.TokenizerFile);

        var modelExists = File.Exists(modelPath) && !FileHasher.IsLfsPointerFile(modelPath);
        var tokenizerExists = File.Exists(tokenizerPath) && !FileHasher.IsLfsPointerFile(tokenizerPath);

        if (modelExists && tokenizerExists)
        {
            return new ModelPaths(modelPath, tokenizerPath);
        }

        if (!_autoDownloadEnabled)
        {
            throw new ModelNotFoundException(
                $"Model '{modelInfo.Id}' not found in cache and auto-download is disabled.",
                modelInfo.Id);
        }

        // Ensure directory exists
        _cacheManager.EnsureModelDirectory(modelInfo.Id);

        // Download model file
        if (!modelExists)
        {
            await _downloadClient.DownloadFileAsync(
                modelInfo.Id,
                modelInfo.OnnxFile,
                modelPath,
                progress: progress,
                cancellationToken: cancellationToken);
        }

        // Download tokenizer file
        if (!tokenizerExists)
        {
            await _downloadClient.DownloadFileAsync(
                modelInfo.Id,
                modelInfo.TokenizerFile,
                tokenizerPath,
                progress: progress,
                cancellationToken: cancellationToken);
        }

        // Verify downloads
        if (!File.Exists(modelPath))
        {
            throw new ModelDownloadException($"Model file was not downloaded successfully.", modelInfo.Id);
        }

        if (!File.Exists(tokenizerPath))
        {
            throw new ModelDownloadException($"Tokenizer file was not downloaded successfully.", modelInfo.Id);
        }

        return new ModelPaths(modelPath, tokenizerPath);
    }

    /// <summary>
    /// Gets the local path for a model if it exists in cache.
    /// </summary>
    /// <param name="modelInfo">Model information.</param>
    /// <returns>Model paths if cached, null otherwise.</returns>
    public ModelPaths? GetCachedModel(ModelInfo modelInfo)
    {
        ArgumentNullException.ThrowIfNull(modelInfo);

        var modelPath = _cacheManager.GetModelFilePath(modelInfo.Id, modelInfo.OnnxFile);
        var tokenizerPath = _cacheManager.GetModelFilePath(modelInfo.Id, modelInfo.TokenizerFile);

        if (File.Exists(modelPath) && File.Exists(tokenizerPath))
        {
            return new ModelPaths(modelPath, tokenizerPath);
        }

        return null;
    }

    /// <summary>
    /// Deletes a model from the cache.
    /// </summary>
    /// <param name="modelId">Model identifier.</param>
    public void DeleteModel(string modelId)
    {
        _cacheManager.DeleteModel(modelId);
    }

    /// <summary>
    /// Gets all cached models.
    /// </summary>
    public IEnumerable<(string ModelId, string Revision)> GetCachedModels()
    {
        return _cacheManager.GetCachedModels();
    }

    /// <summary>
    /// Gets the cache directory path.
    /// </summary>
    public string CacheDirectory => _cacheManager.CacheDirectory;

    public void Dispose()
    {
        if (_disposed) return;
        _downloadClient.Dispose();
        _disposed = true;
    }
}

/// <summary>
/// Paths to model files.
/// </summary>
/// <param name="ModelPath">Path to the ONNX model file.</param>
/// <param name="TokenizerPath">Path to the tokenizer configuration file.</param>
public readonly record struct ModelPaths(string ModelPath, string TokenizerPath);

