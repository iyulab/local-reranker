namespace LocalReranker.Exceptions;

/// <summary>
/// Exception thrown when tokenization fails.
/// </summary>
public class TokenizationException : LocalRerankerException
{
    /// <summary>
    /// Gets the input that caused the tokenization failure.
    /// </summary>
    public string? Input { get; }

    /// <summary>
    /// Initializes a new instance of TokenizationException.
    /// </summary>
    /// <param name="message">Error message.</param>
    public TokenizationException(string message)
        : base(message)
    {
    }

    /// <summary>
    /// Initializes a new instance with input information.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="input">The input that caused the failure.</param>
    public TokenizationException(string message, string input)
        : base(message)
    {
        Input = input.Length > 100 ? input[..100] + "..." : input;
    }

    /// <summary>
    /// Initializes a new instance with an inner exception.
    /// </summary>
    /// <param name="message">Error message.</param>
    /// <param name="innerException">Inner exception.</param>
    public TokenizationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
