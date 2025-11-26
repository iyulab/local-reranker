using System.Net;
using System.Net.Http.Headers;
using LocalReranker.Exceptions;

namespace LocalReranker.Infrastructure;

/// <summary>
/// HTTP client for downloading files from HuggingFace Hub.
/// </summary>
internal sealed class HuggingFaceClient : IDisposable
{
    private const string BaseUrl = "https://huggingface.co";
    private const int MaxRetries = 3;
    private const int BufferSize = 81920; // 80KB

    private readonly HttpClient _httpClient;
    private bool _disposed;

    /// <summary>
    /// Initializes a new HuggingFace client.
    /// </summary>
    /// <param name="httpClient">Optional HttpClient instance.</param>
    public HuggingFaceClient(HttpClient? httpClient = null)
    {
        _httpClient = httpClient ?? CreateDefaultHttpClient();
    }

    /// <summary>
    /// Downloads a file from a HuggingFace repository.
    /// </summary>
    /// <param name="repositoryId">Repository ID (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").</param>
    /// <param name="fileName">File name within the repository.</param>
    /// <param name="destinationPath">Local path to save the file.</param>
    /// <param name="revision">Repository revision (default: "main").</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task DownloadFileAsync(
        string repositoryId,
        string fileName,
        string destinationPath,
        string revision = "main",
        IProgress<float>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(repositoryId);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        ArgumentException.ThrowIfNullOrWhiteSpace(destinationPath);

        var url = BuildResolveUrl(repositoryId, fileName, revision);
        var tempPath = destinationPath + ".tmp";
        var directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        Exception? lastException = null;

        for (var attempt = 1; attempt <= MaxRetries; attempt++)
        {
            try
            {
                await DownloadWithProgressAsync(url, tempPath, progress, cancellationToken);

                // Verify it's not a LFS pointer file
                if (FileHasher.IsLfsPointerFile(tempPath))
                {
                    File.Delete(tempPath);
                    throw new ModelDownloadException(
                        $"Downloaded file is a Git LFS pointer. The model '{repositoryId}' may require Git LFS to download properly.",
                        repositoryId);
                }

                // Move temp file to final destination
                if (File.Exists(destinationPath))
                {
                    File.Delete(destinationPath);
                }
                File.Move(tempPath, destinationPath);
                return;
            }
            catch (OperationCanceledException)
            {
                CleanupTempFile(tempPath);
                throw;
            }
            catch (ModelDownloadException)
            {
                throw;
            }
            catch (Exception ex)
            {
                lastException = ex;
                CleanupTempFile(tempPath);

                if (attempt < MaxRetries)
                {
                    var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                    await Task.Delay(delay, cancellationToken);
                }
            }
        }

        throw new ModelDownloadException(
            $"Failed to download '{fileName}' from '{repositoryId}' after {MaxRetries} attempts.",
            repositoryId,
            lastException);
    }

    /// <summary>
    /// Gets the file size from HuggingFace without downloading.
    /// </summary>
    public async Task<long?> GetFileSizeAsync(
        string repositoryId,
        string fileName,
        string revision = "main",
        CancellationToken cancellationToken = default)
    {
        var url = BuildResolveUrl(repositoryId, fileName, revision);

        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Head, url);
            using var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

            if (response.IsSuccessStatusCode)
            {
                return response.Content.Headers.ContentLength;
            }
        }
        catch
        {
            // Ignore errors for size check
        }

        return null;
    }

    private async Task DownloadWithProgressAsync(
        string url,
        string destinationPath,
        IProgress<float>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        if (!response.IsSuccessStatusCode)
        {
            var statusCode = (int)response.StatusCode;
            throw new HttpRequestException($"Failed to download file. Status: {statusCode} ({response.StatusCode})");
        }

        var totalBytes = response.Content.Headers.ContentLength;
        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var fileStream = new FileStream(
            destinationPath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            BufferSize,
            FileOptions.Asynchronous);

        var buffer = new byte[BufferSize];
        long bytesRead = 0;
        int read;

        while ((read = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
            bytesRead += read;

            if (totalBytes > 0 && progress is not null)
            {
                progress.Report((float)bytesRead / totalBytes.Value);
            }
        }

        progress?.Report(1.0f);
    }

    private static string BuildResolveUrl(string repositoryId, string fileName, string revision)
    {
        // Use 'resolve' endpoint which handles LFS and redirects properly
        return $"{BaseUrl}/{repositoryId}/resolve/{revision}/{fileName}";
    }

    private static HttpClient CreateDefaultHttpClient()
    {
        var handler = new HttpClientHandler
        {
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 10,
            AutomaticDecompression = DecompressionMethods.All
        };

        var client = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromMinutes(30)
        };

        client.DefaultRequestHeaders.UserAgent.Add(
            new ProductInfoHeaderValue("LocalReranker", "1.0"));

        return client;
    }

    private static void CleanupTempFile(string tempPath)
    {
        try
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
        catch
        {
            // Ignore cleanup errors
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _httpClient.Dispose();
        _disposed = true;
    }
}
