namespace LocalReranker;

/// <summary>
/// Configuration options for the Reranker.
/// </summary>
public sealed class RerankerOptions
{
    /// <summary>
    /// Gets or sets the model identifier.
    /// <para>Supports:</para>
    /// <list type="bullet">
    /// <item>Preset aliases: "default", "quality", "fast", "multilingual"</item>
    /// <item>HuggingFace model IDs: "cross-encoder/ms-marco-MiniLM-L-6-v2"</item>
    /// <item>Local file paths: "/path/to/model.onnx"</item>
    /// </list>
    /// <para>Default: "default" (ms-marco-MiniLM-L-6-v2)</para>
    /// </summary>
    public string ModelId { get; set; } = "default";

    /// <summary>
    /// Gets or sets the maximum input sequence length.
    /// Longer inputs will be truncated.
    /// <para>Default: null (uses model's default, typically 512)</para>
    /// </summary>
    public int? MaxSequenceLength { get; set; }

    /// <summary>
    /// Gets or sets the custom cache directory for model files.
    /// <para>Default: null (uses HuggingFace standard cache location)</para>
    /// </summary>
    /// <remarks>
    /// Default location follows HuggingFace standard: ~/.cache/huggingface/hub
    /// <para>Environment variables (in priority order):</para>
    /// <list type="bullet">
    /// <item>HF_HUB_CACHE</item>
    /// <item>HF_HOME + "/hub"</item>
    /// <item>XDG_CACHE_HOME + "/huggingface/hub"</item>
    /// </list>
    /// </remarks>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets the execution provider for inference.
    /// <para>Default: Auto (automatically selects the best available provider)</para>
    /// </summary>
    public ExecutionProvider Provider { get; set; } = ExecutionProvider.Auto;

    /// <summary>
    /// Gets or sets whether to disable automatic model download.
    /// When true, throws an exception if the model is not found locally.
    /// <para>Default: false</para>
    /// </summary>
    public bool DisableAutoDownload { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of inference threads.
    /// <para>Default: null (uses Environment.ProcessorCount)</para>
    /// </summary>
    public int? ThreadCount { get; set; }

    /// <summary>
    /// Gets or sets the batch size for processing multiple documents.
    /// Larger values are faster but use more memory.
    /// <para>Default: 32</para>
    /// </summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>
    /// Creates a copy of these options.
    /// </summary>
    public RerankerOptions Clone() => new()
    {
        ModelId = ModelId,
        MaxSequenceLength = MaxSequenceLength,
        CacheDirectory = CacheDirectory,
        Provider = Provider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        BatchSize = BatchSize
    };

    /// <summary>
    /// Gets the default cache directory path following HuggingFace standard.
    /// </summary>
    public static string GetDefaultCacheDirectory()
    {
        // Priority 1: HF_HUB_CACHE
        var hfHubCache = Environment.GetEnvironmentVariable("HF_HUB_CACHE");
        if (!string.IsNullOrWhiteSpace(hfHubCache))
        {
            return hfHubCache;
        }

        // Priority 2: HF_HOME + "/hub"
        var hfHome = Environment.GetEnvironmentVariable("HF_HOME");
        if (!string.IsNullOrWhiteSpace(hfHome))
        {
            return Path.Combine(hfHome, "hub");
        }

        // Priority 3: XDG_CACHE_HOME + "/huggingface/hub"
        var xdgCache = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        if (!string.IsNullOrWhiteSpace(xdgCache))
        {
            return Path.Combine(xdgCache, "huggingface", "hub");
        }

        // Default: ~/.cache/huggingface/hub
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(userProfile, ".cache", "huggingface", "hub");
    }
}

/// <summary>
/// Specifies the execution provider for ONNX Runtime inference.
/// </summary>
public enum ExecutionProvider
{
    /// <summary>
    /// Automatically select the best available provider.
    /// Selection order: CUDA → DirectML (Windows) / CoreML (macOS) → CPU.
    /// This is the recommended default for zero-configuration usage.
    /// </summary>
    Auto,

    /// <summary>
    /// CPU execution (works everywhere).
    /// </summary>
    Cpu,

    /// <summary>
    /// NVIDIA CUDA GPU acceleration.
    /// </summary>
    Cuda,

    /// <summary>
    /// Windows DirectML GPU acceleration (AMD, Intel, NVIDIA).
    /// </summary>
    DirectML,

    /// <summary>
    /// Apple CoreML acceleration.
    /// </summary>
    CoreML
}
