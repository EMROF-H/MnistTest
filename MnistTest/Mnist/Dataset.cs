using System.Buffers;

namespace MnistTest.Mnist;

internal readonly ref struct RentArray<T>(T[] array, int length)
{
    public Span<T> Span => array.AsSpan(0, Length);
    public int Length { get; } = length;

    public static RentArray<T> Create(int length)
    {
        var array = ArrayPool<T>.Shared.Rent(length);
        return new RentArray<T>(array, length);
    }

    public void Dispose() => ArrayPool<T>.Shared.Return(array, clearArray: false);
}

public class CsvDataset
{
    public List<CsvSample> Samples { get; } = new();

    public CsvDataset(string filePath)
    {
        using var reader = new StreamReader(filePath);
        var isHeader = true;
        while (reader.ReadLine() is { } line)
        {
            if (isHeader)
            {
                isHeader = false; // Skip header line
                continue;
            }
            Samples.Add(new(ParseMnistLine(line)));
        }
    }

    private static readonly Lazy<int[]> LazyCommaIndices = new(() => new int[CsvSample.TotalPartCount],
        LazyThreadSafetyMode.ExecutionAndPublication);
    private static int[] CommaIndices => LazyCommaIndices.Value;

    public static (byte Label, byte[,] Pixels) ParseMnistLine(string line)
    {
        ReadOnlySpan<char> span = line;
        using var commaIndices = RentArray<int>.Create(CsvSample.TotalPartCount);
        var commaCount = 0;
        for (var i = 0; i < span.Length && commaCount < commaIndices.Length - 1; i++)
        {
            if (span[i] != ',')
            {
                continue;
            }
            commaIndices.Span[commaCount] = i;
            commaCount++;
        }
        if (commaCount != CsvSample.PixelCount)
        {
            throw new FormatException(
                $"Expected {CsvSample.TotalPartCount} values (1 label + {CsvSample.PixelCount} pixels), but found {commaCount + 1}.");
        }
        commaIndices.Span[^1] = span.Length;

        // Parse label
        var labelSlice = span[..commaIndices.Span[0]];
        var label = byte.Parse(labelSlice);

        // Parse pixels
        var pixels = new byte[CsvSample.Height, CsvSample.Width];
        var index = 0;
        for (var row = 0; row < CsvSample.Height; row++)
        {
            for (var col = 0; col < CsvSample.Width; col++)
            {
                // var index = row * CsvSample.Width + col;
                var start = commaIndices.Span[index] + 1;
                var len = commaIndices.Span[index + 1] - start;
                var value = byte.Parse(span.Slice(start, len));
                pixels[row, col] = value;

                index++;
            }
        }
        return (label, pixels);
    }
}
