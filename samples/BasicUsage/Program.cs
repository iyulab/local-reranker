using LocalReranker;

Console.WriteLine("LocalReranker - Basic Usage Demo");
Console.WriteLine("================================\n");

// Zero-configuration usage - just works!
await using var reranker = new Reranker();

// Sample query and documents
var query = "What is machine learning?";
var documents = new[]
{
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "The weather today is sunny with a high of 75 degrees.",
    "Deep learning uses neural networks with many layers to process complex patterns.",
    "Python is a popular programming language for data science.",
    "Supervised learning involves training models on labeled data.",
    "The stock market closed higher today.",
    "Neural networks are inspired by the structure of the human brain.",
    "Climate change is affecting global weather patterns.",
    "Reinforcement learning trains agents through trial and error.",
    "Coffee is one of the most popular beverages worldwide."
};

Console.WriteLine($"Query: \"{query}\"\n");
Console.WriteLine($"Documents to rerank: {documents.Length}\n");

// First call will download the model (if not cached)
Console.WriteLine("Reranking documents...\n");
var results = await reranker.RerankAsync(query, documents, topK: 5);

Console.WriteLine("Top 5 Results:");
Console.WriteLine("--------------");
foreach (var result in results)
{
    Console.WriteLine($"  [{result.Score:F4}] {result.Document}");
}

Console.WriteLine("\n================================");
Console.WriteLine("Demo completed!");

// Show model info
var modelInfo = reranker.GetModelInfo();
if (modelInfo != null)
{
    Console.WriteLine($"\nModel: {modelInfo.DisplayName}");
    Console.WriteLine($"Max sequence length: {modelInfo.MaxSequenceLength}");
}
