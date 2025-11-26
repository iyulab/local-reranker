namespace LocalReranker.Exceptions;

/// <summary>
/// Exception thrown when a model is not found.
/// </summary>
public class ModelNotFoundException : LocalRerankerException
{
    /// <summary>
    /// Gets the model identifier that was not found.
    /// </summary>
    public string ModelId { get; }

    /// <summary>
    /// Initializes a new instance of ModelNotFoundException.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="modelId">The model identifier.</param>
    public ModelNotFoundException(string message, string modelId)
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
    public ModelNotFoundException(string message, string modelId, Exception innerException)
        : base(message, innerException)
    {
        ModelId = modelId;
    }
}
