namespace MnistTest.Mnist;

public class CsvSample(byte label, byte[,] pixels)
{
    public const int Width = 28;
    public const int Height = 28;
    public const int PixelCount = Width * Height;
    public const int TotalPartCount = CsvSample.PixelCount + 1;

    public byte Label { get; } = label;
    public byte[,] Pixels { get; } = pixels;
    
    public CsvSample((byte Label, byte[,] Pixels) tuple) : this(tuple.Label, tuple.Pixels) { }
}
