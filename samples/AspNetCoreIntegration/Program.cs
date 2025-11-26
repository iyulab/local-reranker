using LocalReranker;

var builder = WebApplication.CreateBuilder(args);

// Register Reranker as a singleton (recommended for performance)
// The reranker holds the model in memory, so a single instance should be shared
builder.Services.AddSingleton<IReranker>(sp =>
{
    var reranker = new Reranker(new RerankerOptions
    {
        // Use the default model (ms-marco-MiniLM-L-6-v2)
        ModelId = "default",
        // Optional: Customize for production
        BatchSize = 32
    });

    // Optional: Warmup the model on startup
    // This downloads the model (if needed) and loads it into memory
    // Comment this out if you prefer lazy initialization
    reranker.WarmupAsync().Wait();

    return reranker;
});

// Add OpenAPI support for testing
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Enable Swagger in development
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// Sample documents (in a real app, these would come from a database/search engine)
var sampleDocuments = new Dictionary<string, string>
{
    ["doc1"] = "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    ["doc2"] = "Deep learning uses neural networks with many layers to process complex patterns in data.",
    ["doc3"] = "Natural language processing allows computers to understand and generate human language.",
    ["doc4"] = "Python is a popular programming language for data science and machine learning applications.",
    ["doc5"] = "The stock market is influenced by economic indicators and investor sentiment.",
    ["doc6"] = "Cloud computing provides on-demand computing resources over the internet.",
    ["doc7"] = "Reinforcement learning trains agents to make decisions through trial and error.",
    ["doc8"] = "Computer vision enables machines to interpret and understand visual information.",
    ["doc9"] = "Transformers have revolutionized natural language processing with attention mechanisms.",
    ["doc10"] = "Big data analytics involves examining large datasets to uncover patterns and insights."
};

// Health check endpoint
app.MapGet("/health", () => Results.Ok(new { status = "healthy" }))
    .WithName("HealthCheck")
    .WithOpenApi();

// Get model info endpoint
app.MapGet("/model", (IReranker reranker) =>
{
    var info = reranker.GetModelInfo();
    if (info == null)
        return Results.NotFound("Model not loaded yet");

    return Results.Ok(new
    {
        id = info.Id,
        displayName = info.DisplayName,
        maxSequenceLength = info.MaxSequenceLength,
        sizeMB = info.SizeMB
    });
})
.WithName("GetModelInfo")
.WithOpenApi();

// Search and rerank endpoint
app.MapPost("/search", async (SearchRequest request, IReranker reranker) =>
{
    if (string.IsNullOrWhiteSpace(request.Query))
        return Results.BadRequest("Query is required");

    // Get all documents (in a real app, you'd first retrieve candidates from a search engine)
    var documents = sampleDocuments.Values.ToArray();
    var docIds = sampleDocuments.Keys.ToArray();

    // Rerank documents by semantic relevance to the query
    var results = await reranker.RerankAsync(request.Query, documents, request.TopK ?? 5);

    // Map results back to document IDs
    var response = results.Select(r => new SearchResult
    {
        DocumentId = docIds[r.OriginalIndex],
        Content = r.Document,
        Score = r.Score
    }).ToList();

    return Results.Ok(new SearchResponse
    {
        Query = request.Query,
        Results = response
    });
})
.WithName("Search")
.WithOpenApi();

// Batch rerank endpoint (for multiple queries)
app.MapPost("/batch-search", async (BatchSearchRequest request, IReranker reranker) =>
{
    if (request.Queries == null || request.Queries.Count == 0)
        return Results.BadRequest("Queries are required");

    var documents = sampleDocuments.Values.ToList();
    var docIds = sampleDocuments.Keys.ToArray();

    // Create document sets (same documents for each query in this example)
    var documentSets = request.Queries.Select(_ => documents).ToList();

    // Batch rerank all queries
    var batchResults = await reranker.RerankBatchAsync(
        request.Queries,
        documentSets.Select(d => d.AsEnumerable()),
        request.TopK ?? 5);

    // Map results
    var responses = new List<SearchResponse>();
    for (var i = 0; i < request.Queries.Count; i++)
    {
        var queryResults = batchResults[i].Select(r => new SearchResult
        {
            DocumentId = docIds[r.OriginalIndex],
            Content = r.Document,
            Score = r.Score
        }).ToList();

        responses.Add(new SearchResponse
        {
            Query = request.Queries[i],
            Results = queryResults
        });
    }

    return Results.Ok(responses);
})
.WithName("BatchSearch")
.WithOpenApi();

Console.WriteLine("LocalReranker ASP.NET Core Integration Demo");
Console.WriteLine("==========================================");
Console.WriteLine("Swagger UI: http://localhost:5000/swagger");
Console.WriteLine();

app.Run("http://localhost:5000");

// Request/Response models
record SearchRequest(string Query, int? TopK);
record BatchSearchRequest(List<string> Queries, int? TopK);
record SearchResult
{
    public required string DocumentId { get; init; }
    public required string Content { get; init; }
    public required float Score { get; init; }
}
record SearchResponse
{
    public required string Query { get; init; }
    public required List<SearchResult> Results { get; init; }
}
