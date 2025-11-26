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
    /// <para>Default: null (platform-specific default location)</para>
    /// </summary>
    /// <remarks>
    /// Default locations:
    /// <list type="bullet">
    /// <item>Windows: %LOCALAPPDATA%\LocalReranker\models</item>
    /// <item>Linux: ~/.cache/localreranker/models</item>
    /// <item>macOS: ~/Library/Caches/LocalReranker/models</item>
    /// </list>
    /// Can also be set via LOCALRERANKER_CACHE_DIR environment variable.
    /// </remarks>
    public string? CacheDirectory { get; set; }

    /// <summary>
    /// Gets or sets whether to use GPU acceleration if available.
    /// Falls back to CPU if GPU is not available.
    /// <para>Default: false</para>
    /// </summary>
    public bool UseGpu { get; set; } = false;

    /// <summary>
    /// Gets or sets the preferred GPU provider when UseGpu is true.
    /// <para>Default: Auto (tries CUDA, then DirectML, then CPU)</para>
    /// </summary>
    public GpuProvider GpuProvider { get; set; } = GpuProvider.Auto;

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
        UseGpu = UseGpu,
        GpuProvider = GpuProvider,
        DisableAutoDownload = DisableAutoDownload,
        ThreadCount = ThreadCount,
        BatchSize = BatchSize
    };
}

/// <summary>
/// GPU provider options for hardware acceleration.
/// </summary>
public enum GpuProvider
{
    /// <summary>
    /// Automatically select the best available GPU provider.
    /// Tries CUDA → DirectML → CPU.
    /// </summary>
    Auto,

    /// <summary>
    /// Use NVIDIA CUDA (requires NVIDIA GPU and CUDA toolkit).
    /// </summary>
    Cuda,

    /// <summary>
    /// Use DirectML (Windows only, supports any DirectX 12 GPU).
    /// </summary>
    DirectML,

    /// <summary>
    /// Use Apple CoreML (macOS/iOS with Apple Silicon).
    /// </summary>
    CoreML,

    /// <summary>
    /// Force CPU execution (no GPU).
    /// </summary>
    Cpu
}
