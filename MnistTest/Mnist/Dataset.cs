using System.Buffers;

namespace MnistTest.Mnist;

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

        var nextComma = span.IndexOf(',');
        if (nextComma <= 0)
        {
            throw new FormatException("Invalid CSV: missing label field.");
        }

        var label = byte.Parse(span[..nextComma]);
        span = span[(nextComma + 1)..]; // skip label + comma

        // --- Parse pixels ---
        var pixels = new byte[CsvSample.Height, CsvSample.Width];
        var fieldCount = 1; // label counted

        for (var row = 0; row < CsvSample.Height; row++)
        {
            for (var col = 0; col < CsvSample.Width; col++)
            {
                if (span.IsEmpty)
                {
                    throw new FormatException($"Too few fields: expected {CsvSample.TotalPartCount}, found {fieldCount}.");
                }

                nextComma = span.IndexOf(',');
                ReadOnlySpan<char> token;

                if (nextComma >= 0)
                {
                    token = span[..nextComma];
                    span = span[(nextComma + 1)..];
                }
                else
                {
                    // Last pixel field, no comma
                    token = span;
                    span = default;
                }

                pixels[row, col] = byte.Parse(token);
                fieldCount++;
            }
        }

        if (!span.IsEmpty)
        {
            throw new FormatException($"Too many fields: expected {CsvSample.TotalPartCount}, found more.");
        }

        return (label, pixels);
    }
}
