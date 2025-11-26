using System.Buffers;
using LocalReranker.Exceptions;
using LocalReranker.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalReranker.Core;

/// <summary>
/// Handles ONNX Runtime inference for cross-encoder models.
/// </summary>
internal sealed class CrossEncoderInference : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string[] _inputNames;
    private readonly string _outputName;
    private readonly OutputShape _outputShape;
    private readonly bool _hasTokenTypeIds;
    private bool _disposed;

    /// <summary>
    /// Gets the ONNX Runtime session for diagnostics.
    /// </summary>
    public InferenceSession Session => _session;

    private CrossEncoderInference(
        InferenceSession session,
        string[] inputNames,
        string outputName,
        OutputShape outputShape,
        bool hasTokenTypeIds)
    {
        _session = session;
        _inputNames = inputNames;
        _outputName = outputName;
        _outputShape = outputShape;
        _hasTokenTypeIds = hasTokenTypeIds;
    }

    /// <summary>
    /// Creates an inference engine from an ONNX model file.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model.</param>
    /// <param name="modelInfo">Model information.</param>
    /// <param name="useGpu">Whether to use GPU acceleration.</param>
    /// <param name="threadCount">Number of inference threads.</param>
    /// <returns>Configured inference engine.</returns>
    public static CrossEncoderInference Create(
        string modelPath,
        ModelInfo modelInfo,
        bool useGpu = false,
        int? threadCount = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found: {modelPath}", modelPath);
        }

        var sessionOptions = CreateSessionOptions(useGpu, threadCount);

        try
        {
            var session = new InferenceSession(modelPath, sessionOptions);

            // Detect input names
            var inputNames = session.InputMetadata.Keys.ToArray();
            var hasTokenTypeIds = inputNames.Contains("token_type_ids");

            // Detect output name
            var outputName = session.OutputMetadata.Keys.First();

            return new CrossEncoderInference(
                session,
                inputNames,
                outputName,
                modelInfo.OutputShape,
                hasTokenTypeIds);
        }
        catch (Exception ex)
        {
            throw new InferenceException($"Failed to load ONNX model from {modelPath}", ex);
        }
    }

    /// <summary>
    /// Runs inference on a batch of encoded inputs.
    /// </summary>
    /// <param name="batch">Encoded batch of query-document pairs.</param>
    /// <returns>Array of relevance scores (0-1).</returns>
    public float[] Infer(EncodedBatch batch)
    {
        var inputIds = CreateTensor(batch.InputIds, batch.BatchSize, batch.SequenceLength);
        var attentionMask = CreateTensor(batch.AttentionMask, batch.BatchSize, batch.SequenceLength);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = CreateTensor(batch.TokenTypeIds, batch.BatchSize, batch.SequenceLength);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds));
        }

        try
        {
            using var results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            return ExtractScores(outputTensor, batch.BatchSize);
        }
        catch (Exception ex)
        {
            throw new InferenceException("Model inference failed", ex);
        }
    }

    /// <summary>
    /// Runs inference on a single query-document pair.
    /// </summary>
    /// <param name="encoded">Encoded input.</param>
    /// <returns>Relevance score (0-1).</returns>
    public float InferSingle(EncodedInput encoded)
    {
        var inputIds = CreateTensor(encoded.InputIds, 1, encoded.InputIds.Length);
        var attentionMask = CreateTensor(encoded.AttentionMask, 1, encoded.AttentionMask.Length);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };

        if (_hasTokenTypeIds)
        {
            var tokenTypeIds = CreateTensor(encoded.TokenTypeIds, 1, encoded.TokenTypeIds.Length);
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds));
        }

        try
        {
            using var results = _session.Run(inputs);
            var outputTensor = results.First().AsTensor<float>();

            return ExtractScores(outputTensor, 1)[0];
        }
        catch (Exception ex)
        {
            throw new InferenceException("Model inference failed", ex);
        }
    }

    private float[] ExtractScores(Tensor<float> outputTensor, int batchSize)
    {
        var scores = new float[batchSize];
        var dimensions = outputTensor.Dimensions.ToArray();

        switch (_outputShape)
        {
            case OutputShape.SingleLogit:
                // Shape: [batch_size, 1] or [batch_size]
                for (var i = 0; i < batchSize; i++)
                {
                    var logit = dimensions.Length == 1
                        ? outputTensor[i]
                        : outputTensor[i, 0];
                    scores[i] = ScoreNormalizer.Sigmoid(logit);
                }
                break;

            case OutputShape.BinaryClassification:
                // Shape: [batch_size, 2] - use softmax on positive class
                for (var i = 0; i < batchSize; i++)
                {
                    var logit0 = outputTensor[i, 0];
                    var logit1 = outputTensor[i, 1];
                    scores[i] = ScoreNormalizer.SoftmaxPositive(logit0, logit1);
                }
                break;

            case OutputShape.FlatLogit:
                // Shape: [batch_size]
                for (var i = 0; i < batchSize; i++)
                {
                    scores[i] = ScoreNormalizer.Sigmoid(outputTensor[i]);
                }
                break;
        }

        return scores;
    }

    private static DenseTensor<long> CreateTensor(long[] data, int batchSize, int sequenceLength)
    {
        return new DenseTensor<long>(data, [batchSize, sequenceLength]);
    }

    private static SessionOptions CreateSessionOptions(bool useGpu, int? threadCount)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_PARALLEL
        };

        // Set thread count
        var threads = threadCount ?? Environment.ProcessorCount;
        options.IntraOpNumThreads = threads;
        options.InterOpNumThreads = Math.Max(1, threads / 2);

        // Try to use GPU if requested
        if (useGpu)
        {
            TryAddGpuProvider(options);
        }

        return options;
    }

    private static void TryAddGpuProvider(SessionOptions options)
    {
        // Try CUDA first (NVIDIA)
        try
        {
            options.AppendExecutionProvider_CUDA();
            return;
        }
        catch
        {
            // CUDA not available
        }

        // Try DirectML (Windows)
        if (OperatingSystem.IsWindows())
        {
            try
            {
                options.AppendExecutionProvider_DML();
                return;
            }
            catch
            {
                // DirectML not available
            }
        }

        // Fall back to CPU (no action needed, it's the default)
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }
}
