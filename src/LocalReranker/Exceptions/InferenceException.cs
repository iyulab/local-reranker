namespace LocalReranker.Exceptions;

/// <summary>
/// Exception thrown when model inference fails.
/// </summary>
public class InferenceException : LocalRerankerException
{
    /// <summary>
    /// Initializes a new instance of InferenceException.
    /// </summary>
    /// <param name="message">Error message.</param>
    public InferenceException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance with an inner exception.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="innerException">Inner exception.</param>
    public InferenceException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
