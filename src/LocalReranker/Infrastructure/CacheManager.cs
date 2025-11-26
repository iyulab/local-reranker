namespace LocalReranker.Infrastructure;

/// <summary>
/// Manages local cache directories for model storage.
/// </summary>
internal sealed class CacheManager
{
    private const string CacheEnvVar = "LOCALRERANKER_CACHE_DIR";
    private const string AppName = "LocalReranker";
    private const string ModelsFolder = "models";

    private readonly string _cacheDirectory;

    /// <summary>
    /// Initializes a new instance with optional custom cache directory.
    /// </summary>
    /// <param name="customCacheDirectory">Custom cache directory path, or null to use default.</param>
    public CacheManager(string? customCacheDirectory = null)
    {
        _cacheDirectory = ResolveDirectory(customCacheDirectory);
    }

    /// <summary>
    /// Gets the root cache directory.
    /// </summary>
    public string CacheDirectory => _cacheDirectory;

    /// <summary>
    /// Gets the full path for a model's cache directory.
    /// </summary>
    /// <param name="modelId">The model identifier (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").</param>
    /// <param name="revision">The model revision (default: "main").</param>
    /// <returns>Full path to the model directory.</returns>
    public string GetModelDirectory(string modelId, string revision = "main")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);

        var sanitizedModelId = SanitizeModelId(modelId);
        return Path.Combine(_cacheDirectory, sanitizedModelId, revision);
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
        var targetPath = revision is null
            ? Path.Combine(_cacheDirectory, sanitizedModelId)
            : Path.Combine(_cacheDirectory, sanitizedModelId, revision);

        if (Directory.Exists(targetPath))
        {
            Directory.Delete(targetPath, recursive: true);
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

        foreach (var modelDir in Directory.GetDirectories(_cacheDirectory))
        {
            var modelId = Path.GetFileName(modelDir).Replace("--", "/");
            foreach (var revisionDir in Directory.GetDirectories(modelDir))
            {
                var revision = Path.GetFileName(revisionDir);
                yield return (modelId, revision);
            }
        }
    }

    private static string ResolveDirectory(string? customPath)
    {
        // Priority: Custom path > Environment variable > Platform default
        if (!string.IsNullOrWhiteSpace(customPath))
        {
            return customPath;
        }

        var envPath = Environment.GetEnvironmentVariable(CacheEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath))
        {
            return envPath;
        }

        return GetPlatformDefaultPath();
    }

    private static string GetPlatformDefaultPath()
    {
        if (OperatingSystem.IsWindows())
        {
            // Windows: %LOCALAPPDATA%\LocalReranker\models
            var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(localAppData, AppName, ModelsFolder);
        }

        if (OperatingSystem.IsMacOS())
        {
            // macOS: ~/Library/Caches/LocalReranker/models
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(home, "Library", "Caches", AppName, ModelsFolder);
        }

        // Linux and others: ~/.cache/localreranker/models
        var xdgCache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (!string.IsNullOrWhiteSpace(xdgCache))
        {
            return Path.Combine(xdgCache, AppName.ToLowerInvariant(), ModelsFolder);
        }

        var homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(homeDir, ".cache", AppName.ToLowerInvariant(), ModelsFolder);
    }

    private static string SanitizeModelId(string modelId)
    {
        // Replace path separators with double dashes for safe directory names
        return modelId.Replace("/", "--").Replace("\\", "--");
    }
}
