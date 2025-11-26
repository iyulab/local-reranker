using System.Security.Cryptography;

namespace LocalReranker.Infrastructure;

/// <summary>
/// Provides file hashing utilities for integrity verification.
/// </summary>
internal static class FileHasher
{
    private const int BufferSize = 81920; // 80KB buffer for streaming

    /// <summary>
    /// Computes the SHA256 hash of a file asynchronously.
    /// </summary>
    /// <param name="filePath">Path to the file to hash.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Lowercase hexadecimal hash string.</returns>
    public static async Task<string> ComputeSha256Async(
        string filePath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("File not found for hashing.", filePath);
        }

        await using var stream = new FileStream(
            filePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            BufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        var hashBytes = await SHA256.HashDataAsync(stream, cancellationToken);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    /// <summary>
    /// Computes the SHA256 hash of a file synchronously.
    /// </summary>
    /// <param name="filePath">Path to the file to hash.</param>
    /// <returns>Lowercase hexadecimal hash string.</returns>
    public static string ComputeSha256(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("File not found for hashing.", filePath);
        }

        using var stream = new FileStream(
            filePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            BufferSize,
            FileOptions.SequentialScan);

        var hashBytes = SHA256.HashData(stream);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    /// <summary>
    /// Verifies that a file matches an expected hash.
    /// </summary>
    /// <param name="filePath">Path to the file to verify.</param>
    /// <param name="expectedHash">Expected SHA256 hash (hexadecimal string).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if the hash matches, false otherwise.</returns>
    public static async Task<bool> VerifyHashAsync(
        string filePath,
        string expectedHash,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(expectedHash);

        var actualHash = await ComputeSha256Async(filePath, cancellationToken);
        return string.Equals(actualHash, expectedHash.ToLowerInvariant(), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Checks if a file appears to be a Git LFS pointer file.
    /// </summary>
    /// <param name="filePath">Path to the file to check.</param>
    /// <returns>True if the file is likely a Git LFS pointer.</returns>
    public static bool IsLfsPointerFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            return false;
        }

        var fileInfo = new FileInfo(filePath);

        // LFS pointer files are typically very small (< 200 bytes)
        if (fileInfo.Length > 500)
        {
            return false;
        }

        try
        {
            var content = File.ReadAllText(filePath);
            return content.StartsWith("version https://git-lfs.github.com/spec/", StringComparison.Ordinal);
        }
        catch
        {
            return false;
        }
    }
}
