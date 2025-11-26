namespace LocalReranker.Exceptions;

/// <summary>
/// Base exception for all LocalReranker errors.
/// </summary>
public class LocalRerankerException : Exception
{
    /// <summary>
    /// Initializes a new instance of LocalRerankerException.
    /// </summary>
    public LocalRerankerException()
    {
    }

    /// <summary>
    /// Initializes a new instance with a message.
    /// </summary>
    /// <param name="message">Error message.</param>
    public LocalRerankerException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance with a message and inner exception.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="innerException">Inner exception.</param>
    public LocalRerankerException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
