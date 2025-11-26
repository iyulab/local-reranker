namespace LocalReranker.Exceptions;

/// <summary>
/// Exception thrown when model download fails.
/// </summary>
public class ModelDownloadException : LocalRerankerException
{
    /// <summary>
    /// Gets the model identifier that failed to download.
    /// </summary>
    public string ModelId { get; }

    /// <summary>
    /// Initializes a new instance of ModelDownloadException.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="modelId">The model identifier.</param>
    public ModelDownloadException(string message, string modelId)
        : base(message)
    {
        ModelId = modelId;
    }

    /// <summary>
    /// Initializes a new instance with an inner exception.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="modelId">The model identifier.</param>
    /// <param name="innerException">Inner exception.</param>
    public ModelDownloadException(string message, string modelId, Exception? innerException)
        : base(message, innerException!)
    {
        ModelId = modelId;
    }
}
