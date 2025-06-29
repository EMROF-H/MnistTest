using NeuralNetwork.Layers;

namespace NeuralNetwork;

public class NeuralNetwork
{
    private readonly List<ILayer> layers = new();

    public IReadOnlyList<ILayer> Layers => layers;

    public void AddLayer(ILayer layer)
    {
        layers.Add(layer);
    }

    public void Initialize()
    {
        foreach (var layer in layers)
        {
            if (layer is IParameterizedLayer initLayer)
            {
                initLayer.Initialize();
            }
        }
    }

    public void LoadParameters(Stream stream)
    {
        using var reader = new BinaryReader(stream);
        foreach (var layer in layers)
        {
            if (layer is IParameterizedLayer paramLayer)
            {
                paramLayer.Load(reader);
            }
        }
    }

    public void Train(Vector input, Vector expected, double learningRate = 0.01)
    {
        var context = new ForwardContext();
        var current = input;
        foreach (var layer in layers)
        {
            current = layer.Forward(current, context);
        }

        var grad = current - expected;
        for (int i = layers.Count - 1; i >= 0; i--)
        {
            grad = layers[i].Backward(grad, context, learningRate);
        }
    }

    public Vector Predict(Vector input)
    {
        var current = input;
        foreach (var layer in layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    public void SaveParameters(Stream stream)
    {
        using var writer = new BinaryWriter(stream);
        foreach (var layer in layers)
        {
            if (layer is IParameterizedLayer paramLayer)
            {
                paramLayer.Save(writer);
            }
        }
    }
}
