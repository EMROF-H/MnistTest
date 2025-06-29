namespace NeuralNetwork.Layers;

public interface ILayer
{
    Vector Forward(Vector input);
    Vector Forward(Vector input, ForwardContext context);
    Vector Backward(Vector gradient, ForwardContext context, double learningRate);
}

public interface IParameterizedLayer : ILayer
{
    void Initialize();
    void Load(BinaryReader reader);
    void Save(BinaryWriter writer);
}
