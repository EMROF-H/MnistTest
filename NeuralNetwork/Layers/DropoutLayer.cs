using MathNet.Numerics.Distributions;

namespace NeuralNetwork.Layers;

public class DropoutLayer(double dropoutRate) : ILayer
{
    private Vector? lastMask;
    private readonly double keepRate = 1.0 - dropoutRate;
    private readonly Random random = new();

    public int InputSize { get; private set; }
    public int OutputSize { get; private set; }

    public Vector Forward(Vector input)
    {
        // 推理模式，不使用 Dropout
        return input;
    }

    public Vector Forward(Vector input, ForwardContext context)
    {
        var mask = Vector.Build.Dense(input.Count, i => Bernoulli.Sample(random, keepRate) == 1 ? 1.0 : 0.0);
        lastMask = mask;

        var dropped = input.PointwiseMultiply(mask) / keepRate; // 保持期望值不变
        context.Inputs[this] = dropped;
        return dropped;
    }

    public Vector Backward(Vector gradient, ForwardContext context, double learningRate)
    {
        if (lastMask == null)
            throw new InvalidOperationException("Dropout mask not set. Must call Forward with context before Backward.");

        return gradient.PointwiseMultiply(lastMask) / keepRate;
    }

    public void Initialize() { }

    public void Load(BinaryReader reader) { }

    public void Save(BinaryWriter writer) { }
}
