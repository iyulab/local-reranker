namespace LocalReranker.Infrastructure;

/// <summary>
/// Manages local cache directories for model storage following HuggingFace standard.
/// </summary>
internal sealed class CacheManager
{
    private readonly string _cacheDirectory;

    /// <summary>
    /// Initializes a new instance with optional custom cache directory.
    /// </summary>
    /// <param name="customCacheDirectory">Custom cache directory path, or null to use HuggingFace default.</param>
    public CacheManager(string? customCacheDirectory = null)
    {
        _cacheDirectory = customCacheDirectory ?? RerankerOptions.GetDefaultCacheDirectory();
    }

    /// <summary>
    /// Gets the root cache directory.
    /// </summary>
    public string CacheDirectory => _cacheDirectory;

    /// <summary>
    /// Gets the full path for a model's cache directory following HuggingFace structure.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").</param>
    /// <param name="revision">The model revision (default: "main").</param>
    /// <returns>Full path to the model directory.</returns>
    /// <remarks>
    /// Uses HuggingFace cache structure: models--{org}--{model}/snapshots/{revision}
    /// </remarks>
    public string GetModelDirectory(string modelId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);

        // HuggingFace cache structure: models--{org}--{model}/snapshots/{revision}
        var sanitizedModelId = SanitizeModelId(modelId);
        return Path.Combine(_cacheDirectory, $"models--{sanitizedModelId}", "snapshots", revision);
    }

    /// <summary>
    /// Gets the full path for a specific file within a model's cache.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="fileName">The file name within the model directory.</param>
    /// <param name="revision">The model revision (default: "main").</param>
    /// <returns>Full path to the file.</returns>
    public string GetModelFilePath(string modelId, string fileName, string revision = "main")
    {
        var modelDir = GetModelDirectory(modelId, revision);
        return Path.Combine(modelDir, fileName);
    }

    /// <summary>
    /// Ensures the model directory exists.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="revision">The model revision.</param>
    /// <returns>The path to the created directory.</returns>
    public string EnsureModelDirectory(string modelId, string revision = "main")
    {
        var dir = GetModelDirectory(modelId, revision);
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Checks if a model file exists in the cache.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="fileName">The file name to check.</param>
    /// <param name="revision">The model revision.</param>
    /// <returns>True if the file exists.</returns>
    public bool ModelFileExists(string modelId, string fileName, string revision = "main")
    {
        var filePath = GetModelFilePath(modelId, fileName, revision);
        return File.Exists(filePath);
    }

    /// <summary>
    /// Deletes a model from the cache.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="revision">The model revision, or null to delete all revisions.</param>
    public void DeleteModel(string modelId, string? revision = null)
    {
        var sanitizedModelId = SanitizeModelId(modelId);
        var modelDir = Path.Combine(_cacheDirectory, $"models--{sanitizedModelId}");

        if (revision is null)
        {
            // Delete entire model directory
            if (Directory.Exists(modelDir))
            {
                Directory.Delete(modelDir, recursive: true);
            }
        }
        else
        {
            // Delete specific revision
            var revisionPath = Path.Combine(modelDir, "snapshots", revision);
            if (Directory.Exists(revisionPath))
            {
                Directory.Delete(revisionPath, recursive: true);
            }
        }
    }

    /// <summary>
    /// Gets all cached models.
    /// </summary>
    /// <returns>Collection of model IDs and their revisions.</returns>
    public IEnumerable<(string ModelId, string Revision)> GetCachedModels()
    {
        if (!Directory.Exists(_cacheDirectory))
        {
            yield break;
        }

        // HuggingFace structure: models--{org}--{model}/snapshots/{revision}
        foreach (var modelDir in Directory.GetDirectories(_cacheDirectory, "models--*"))
        {
            var dirName = Path.GetFileName(modelDir);
            // Remove "models--" prefix and restore slashes
            var modelId = dirName[8..].Replace("--", "/");

            var snapshotsDir = Path.Combine(modelDir, "snapshots");
            if (!Directory.Exists(snapshotsDir)) continue;

            foreach (var revisionDir in Directory.GetDirectories(snapshotsDir))
            {
                var revision = Path.GetFileName(revisionDir);
                yield return (modelId, revision);
            }
        }
    }

    private static string SanitizeModelId(string modelId)
    {
        // Replace path separators with double dashes for safe directory names
        return modelId.Replace("/", "--").Replace("\\", "--");
    }
}
