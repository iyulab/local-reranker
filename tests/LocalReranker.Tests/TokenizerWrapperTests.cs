using FluentAssertions;
using LocalReranker.Core;
using Xunit;

namespace LocalReranker.Tests;

/// <summary>
/// Tests for TokenizerWrapper functionality.
/// Note: These tests require tokenizer files which may not be available in CI.
/// They are designed to test the encoding logic when files are present.
/// </summary>
public class TokenizerWrapperTests
{
    [Fact]
    public void EncodedInput_ShouldHaveCorrectStructure()
    {
        // Arrange
        var inputIds = new long[] { 101, 2054, 2003, 3698, 102, 3698, 2003, 2204, 102, 0 };
        var attentionMask = new long[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 };
        var tokenTypeIds = new long[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 0 };

        // Act
        var encoded = new EncodedInput(inputIds, attentionMask, tokenTypeIds, 9);

        // Assert
        encoded.InputIds.Should().HaveCount(10);
        encoded.AttentionMask.Should().HaveCount(10);
        encoded.TokenTypeIds.Should().HaveCount(10);
        encoded.Length.Should().Be(9);
    }

    [Fact]
    public void EncodedInput_FirstToken_ShouldBeCLS()
    {
        // Arrange - typical BERT [CLS] token ID is 101
        var inputIds = new long[] { 101, 2054, 2003, 102 };
        var attentionMask = new long[] { 1, 1, 1, 1 };
        var tokenTypeIds = new long[] { 0, 0, 0, 0 };

        // Act
        var encoded = new EncodedInput(inputIds, attentionMask, tokenTypeIds, 4);

        // Assert
        encoded.InputIds[0].Should().Be(101); // [CLS]
    }

    [Fact]
    public void EncodedBatch_ShouldFlattenCorrectly()
    {
        // Arrange
        const int batchSize = 2;
        const int seqLength = 4;
        var batch = new EncodedBatch(batchSize, seqLength);

        var input1 = new EncodedInput(
            new long[] { 101, 1, 2, 102 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 0, 0, 0, 0 },
            4);

        var input2 = new EncodedInput(
            new long[] { 101, 3, 4, 102 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 0, 0, 0, 0 },
            4);

        // Act
        batch.SetInput(0, input1);
        batch.SetInput(1, input2);

        // Assert
        batch.BatchSize.Should().Be(2);
        batch.SequenceLength.Should().Be(4);
        batch.InputIds.Should().HaveCount(8); // 2 * 4

        // First batch
        batch.InputIds[0].Should().Be(101);
        batch.InputIds[1].Should().Be(1);
        batch.InputIds[2].Should().Be(2);
        batch.InputIds[3].Should().Be(102);

        // Second batch
        batch.InputIds[4].Should().Be(101);
        batch.InputIds[5].Should().Be(3);
        batch.InputIds[6].Should().Be(4);
        batch.InputIds[7].Should().Be(102);
    }

    [Fact]
    public void EncodedBatch_AttentionMask_ShouldBeSetCorrectly()
    {
        // Arrange
        var batch = new EncodedBatch(1, 6);
        var input = new EncodedInput(
            new long[] { 101, 2054, 102, 0, 0, 0 },
            new long[] { 1, 1, 1, 0, 0, 0 },
            new long[] { 0, 0, 0, 0, 0, 0 },
            3);

        // Act
        batch.SetInput(0, input);

        // Assert
        batch.AttentionMask[0].Should().Be(1); // Real token
        batch.AttentionMask[1].Should().Be(1); // Real token
        batch.AttentionMask[2].Should().Be(1); // Real token
        batch.AttentionMask[3].Should().Be(0); // Padding
        batch.AttentionMask[4].Should().Be(0); // Padding
        batch.AttentionMask[5].Should().Be(0); // Padding
    }

    [Fact]
    public void EncodedBatch_TokenTypeIds_ShouldDistinguishQueryAndDocument()
    {
        // Arrange - Format: [CLS] query [SEP] document [SEP]
        var batch = new EncodedBatch(1, 8);
        var input = new EncodedInput(
            new long[] { 101, 2054, 102, 2003, 2204, 102, 0, 0 },
            new long[] { 1, 1, 1, 1, 1, 1, 0, 0 },
            new long[] { 0, 0, 0, 1, 1, 1, 0, 0 }, // 0 for query segment, 1 for document
            6);

        // Act
        batch.SetInput(0, input);

        // Assert
        batch.TokenTypeIds[0].Should().Be(0); // [CLS] - query segment
        batch.TokenTypeIds[1].Should().Be(0); // query token
        batch.TokenTypeIds[2].Should().Be(0); // [SEP] - query segment
        batch.TokenTypeIds[3].Should().Be(1); // document token
        batch.TokenTypeIds[4].Should().Be(1); // document token
        batch.TokenTypeIds[5].Should().Be(1); // [SEP] - document segment
    }

    [Fact]
    public void EncodedBatch_MultipleBatches_ShouldMaintainIndependence()
    {
        // Arrange
        var batch = new EncodedBatch(3, 4);

        // Act
        batch.SetInput(0, new EncodedInput(
            new long[] { 1, 1, 1, 1 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 0, 0, 0, 0 }, 4));

        batch.SetInput(1, new EncodedInput(
            new long[] { 2, 2, 2, 2 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 0, 0, 0, 0 }, 4));

        batch.SetInput(2, new EncodedInput(
            new long[] { 3, 3, 3, 3 },
            new long[] { 1, 1, 1, 1 },
            new long[] { 0, 0, 0, 0 }, 4));

        // Assert - Each batch should have distinct values
        batch.InputIds[0..4].Should().AllBeEquivalentTo(1L);
        batch.InputIds[4..8].Should().AllBeEquivalentTo(2L);
        batch.InputIds[8..12].Should().AllBeEquivalentTo(3L);
    }
}
