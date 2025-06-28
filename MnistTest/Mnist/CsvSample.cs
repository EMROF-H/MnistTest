namespace MnistTest.Mnist;

public class CsvSample(byte label, byte[,] pixels)
{
    public const string DatasetName = "MNIST";
    
    public const int Width = 28;
    public const int Height = 28;
    public const int PixelCount = Width * Height;
    public const int TotalPartCount = PixelCount + 1;
    public const int MinLineDataLength = 1 + PixelCount * 2;
    public const int MaxLineDataLength = 1 + PixelCount * 4;

    public byte Label { get; } = label;
    public byte[,] Pixels { get; } = pixels;
    
    public CsvSample((byte Label, byte[,] Pixels) tuple) : this(tuple.Label, tuple.Pixels) { }
}
