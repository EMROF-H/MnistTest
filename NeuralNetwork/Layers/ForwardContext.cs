namespace NeuralNetwork.Layers;

public class ForwardContext
{
    public Dictionary<ILayer, Vector> Activations { get; } = new();
    public Dictionary<ILayer, Vector> PreActivations { get; } = new();
    public Dictionary<ILayer, Vector> Inputs { get; } = new();
}
