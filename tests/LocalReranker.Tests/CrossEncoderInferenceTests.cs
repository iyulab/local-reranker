using FluentAssertions;
using LocalReranker.Core;
using LocalReranker.Models;
using Xunit;

namespace LocalReranker.Tests;

/// <summary>
/// Tests for CrossEncoderInference functionality.
/// Note: Tests requiring actual ONNX model files are marked with [Trait].
/// Unit tests for validation logic do not require model files.
/// </summary>
public class CrossEncoderInferenceTests
{
    [Fact]
    public void Create_WithNullPath_ShouldThrowArgumentException()
    {
        // Arrange
        var modelInfo = CreateTestModelInfo();

        // Act
        var act = () => CrossEncoderInference.Create(null!, modelInfo);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Create_WithEmptyPath_ShouldThrowArgumentException()
    {
        // Arrange
        var modelInfo = CreateTestModelInfo();

        // Act
        var act = () => CrossEncoderInference.Create(string.Empty, modelInfo);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Create_WithWhitespacePath_ShouldThrowArgumentException()
    {
        // Arrange
        var modelInfo = CreateTestModelInfo();

        // Act
        var act = () => CrossEncoderInference.Create("   ", modelInfo);

        // Assert
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Create_WithNonExistentPath_ShouldThrowFileNotFoundException()
    {
        // Arrange
        var modelInfo = CreateTestModelInfo();
        var nonExistentPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString(), "model.onnx");

        // Act
        var act = () => CrossEncoderInference.Create(nonExistentPath, modelInfo);

        // Assert
        act.Should().Throw<FileNotFoundException>()
            .WithMessage($"*{nonExistentPath}*");
    }

    [Fact]
    public void ScoreNormalizer_Sigmoid_ShouldReturnZeroPointFiveForZero()
    {
        // Act
        var result = ScoreNormalizer.Sigmoid(0f);

        // Assert
        result.Should().BeApproximately(0.5f, 0.001f);
    }

    [Fact]
    public void ScoreNormalizer_Sigmoid_ShouldReturnNearOneForLargePositive()
    {
        // Act
        var result = ScoreNormalizer.Sigmoid(10f);

        // Assert
        result.Should().BeGreaterThan(0.999f);
    }

    [Fact]
    public void ScoreNormalizer_Sigmoid_ShouldReturnNearZeroForLargeNegative()
    {
        // Act
        var result = ScoreNormalizer.Sigmoid(-10f);

        // Assert
        result.Should().BeLessThan(0.001f);
    }

    [Fact]
    public void ScoreNormalizer_Sigmoid_ShouldHandleExtremePositive()
    {
        // Act - extreme value should not overflow
        var result = ScoreNormalizer.Sigmoid(100f);

        // Assert
        result.Should().Be(1f);
    }

    [Fact]
    public void ScoreNormalizer_Sigmoid_ShouldHandleExtremeNegative()
    {
        // Act - extreme value should not overflow
        var result = ScoreNormalizer.Sigmoid(-100f);

        // Assert
        result.Should().Be(0f);
    }

    [Fact]
    public void ScoreNormalizer_SoftmaxPositive_ShouldReturnHalfForEqualLogits()
    {
        // Act
        var result = ScoreNormalizer.SoftmaxPositive(0f, 0f);

        // Assert
        result.Should().BeApproximately(0.5f, 0.001f);
    }

    [Fact]
    public void ScoreNormalizer_SoftmaxPositive_ShouldReturnHighForPositiveClass()
    {
        // Arrange - positive class (logit1) is much higher
        var result = ScoreNormalizer.SoftmaxPositive(-5f, 5f);

        // Assert
        result.Should().BeGreaterThan(0.99f);
    }

    [Fact]
    public void ScoreNormalizer_SoftmaxPositive_ShouldReturnLowForNegativeClass()
    {
        // Arrange - negative class (logit0) is much higher
        var result = ScoreNormalizer.SoftmaxPositive(5f, -5f);

        // Assert
        result.Should().BeLessThan(0.01f);
    }

    [Fact]
    public void ScoreNormalizer_MinMaxNormalize_EmptyArray_ShouldReturnEmpty()
    {
        // Act
        var result = ScoreNormalizer.MinMaxNormalize(ReadOnlySpan<float>.Empty);

        // Assert
        result.Should().BeEmpty();
    }

    [Fact]
    public void ScoreNormalizer_MinMaxNormalize_SingleElement_ShouldReturnOne()
    {
        // Act
        var result = ScoreNormalizer.MinMaxNormalize([0.5f]);

        // Assert
        result.Should().HaveCount(1);
        result[0].Should().Be(1f);
    }

    [Fact]
    public void ScoreNormalizer_MinMaxNormalize_ShouldNormalizeToZeroOne()
    {
        // Arrange
        var scores = new float[] { 0f, 5f, 10f };

        // Act
        var result = ScoreNormalizer.MinMaxNormalize(scores);

        // Assert
        result.Should().HaveCount(3);
        result[0].Should().BeApproximately(0f, 0.001f); // min
        result[1].Should().BeApproximately(0.5f, 0.001f); // middle
        result[2].Should().BeApproximately(1f, 0.001f); // max
    }

    [Fact]
    public void ScoreNormalizer_MinMaxNormalize_AllSameValues_ShouldReturnHalf()
    {
        // Arrange
        var scores = new float[] { 5f, 5f, 5f };

        // Act
        var result = ScoreNormalizer.MinMaxNormalize(scores);

        // Assert
        result.Should().AllBeEquivalentTo(0.5f);
    }

    [Fact]
    public void ScoreNormalizer_SigmoidInPlace_ShouldModifyArray()
    {
        // Arrange
        var logits = new float[] { 0f, -10f, 10f };

        // Act
        ScoreNormalizer.SigmoidInPlace(logits);

        // Assert
        logits[0].Should().BeApproximately(0.5f, 0.001f);
        logits[1].Should().BeLessThan(0.001f);
        logits[2].Should().BeGreaterThan(0.999f);
    }

    [Fact]
    public void ScoreNormalizer_SigmoidArray_ShouldReturnNewArray()
    {
        // Arrange
        var logits = new float[] { 0f, -10f, 10f };

        // Act
        var result = ScoreNormalizer.Sigmoid(logits.AsSpan());

        // Assert
        result.Should().HaveCount(3);
        result[0].Should().BeApproximately(0.5f, 0.001f);
        result[1].Should().BeLessThan(0.001f);
        result[2].Should().BeGreaterThan(0.999f);

        // Original should be unchanged
        logits[0].Should().Be(0f);
    }

    [Fact]
    public void OutputShape_SingleLogit_ShouldBeDefined()
    {
        // Assert
        OutputShape.SingleLogit.Should().BeDefined();
    }

    [Fact]
    public void OutputShape_BinaryClassification_ShouldBeDefined()
    {
        // Assert
        OutputShape.BinaryClassification.Should().BeDefined();
    }

    [Fact]
    public void OutputShape_FlatLogit_ShouldBeDefined()
    {
        // Assert
        OutputShape.FlatLogit.Should().BeDefined();
    }

    [Fact]
    public void ModelArchitecture_AllTypes_ShouldBeDefined()
    {
        // Assert
        ModelArchitecture.Bert.Should().BeDefined();
        ModelArchitecture.Roberta.Should().BeDefined();
        ModelArchitecture.XlmRoberta.Should().BeDefined();
        ModelArchitecture.JinaBert.Should().BeDefined();
    }

    private static ModelInfo CreateTestModelInfo()
    {
        return new ModelInfo
        {
            Id = "test/test-model",
            Alias = "test",
            DisplayName = "Test Model",
            Parameters = 22_000_000,
            MaxSequenceLength = 512,
            SizeBytes = 90_000_000,
            OnnxFile = "onnx/model.onnx",
            TokenizerFile = "tokenizer.json",
            Description = "Test model for unit tests",
            OutputShape = OutputShape.SingleLogit,
            Architecture = ModelArchitecture.Bert
        };
    }
}
