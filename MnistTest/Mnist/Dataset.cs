using System.Buffers;
using System.IO;

namespace MnistTest.Mnist;

public class CsvDataset
{
    public List<CsvSample> Samples { get; } = new();

    private static readonly char[] Line = new char[CsvSample.MaxLineDataLength];

    public CsvDataset(string filePath)
    {
        using var reader = new StreamReader(filePath);
        var isHeader = true;
        while (ReadLine(reader, Line.AsSpan(), out var charsRead))
        {
            if (isHeader)
            {
                isHeader = false;
                continue;
            }

            var lineSpan = Line.AsSpan(0, charsRead);
            Samples.Add(new(ParseMnistLine(lineSpan)));
        }
    }

    private static bool ReadLine(StreamReader reader, Span<char> buffer, out int charsRead)
    {
        charsRead = 0;
        while (true)
        {
            var ch = reader.Read();
            switch (ch)
            {
                case -1:
                    return charsRead > 0; // EOF
                case '\r':
                    continue; // skip CR
                case '\n':
                    return true; // newline end
            }

            if (charsRead >= buffer.Length)
            {
                throw new InvalidOperationException("Line exceeds buffer length");
            }
            buffer[charsRead++] = (char)ch;
        }
    }

    private static (byte Label, byte[,] Pixels) ParseMnistLine(ReadOnlySpan<char> line)
    {
        var nextComma = line.IndexOf(',');
        if (nextComma <= 0)
        {
            throw new FormatException("Invalid CSV: missing label field.");
        }

        var label = byte.Parse(line[..nextComma]);
        line = line[(nextComma + 1)..]; // skip label + comma

        var pixels = new byte[CsvSample.Height, CsvSample.Width];
        var fieldCount = 1; // label counted

        for (var row = 0; row < CsvSample.Height; row++)
        {
            for (var col = 0; col < CsvSample.Width; col++)
            {
                if (line.IsEmpty)
                {
                    throw new FormatException($"Too few fields: expected {CsvSample.TotalPartCount}, found {fieldCount}.");
                }

                nextComma = line.IndexOf(',');
                ReadOnlySpan<char> token;

                if (nextComma >= 0)
                {
                    token = line[..nextComma];
                    line = line[(nextComma + 1)..];
                }
                else
                {
                    // Last pixel field, no comma
                    token = line;
                    line = default;
                }

                pixels[row, col] = byte.Parse(token);
                fieldCount++;
            }
        }

        if (!line.IsEmpty)
        {
            throw new FormatException($"Too many fields: expected {CsvSample.TotalPartCount}, found more.");
        }

        return (label, pixels);
    }
}
