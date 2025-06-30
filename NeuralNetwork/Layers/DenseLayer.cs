using MathNet.Numerics.Distributions;
using NeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Layers;

public class DenseLayer<TActivation>(int inputSize, int outputSize) : IParameterizedLayer
    where TActivation : IActivationFunction
{
    private Matrix weights = Matrix.Build.Random(outputSize, inputSize);
    private Vector biases = Vector.Build.Dense(outputSize);

    public int InputSize { get; } = inputSize;
    public int OutputSize { get; } = outputSize;

    public Vector Forward(Vector input) =>
        TActivation.Apply(weights * input + biases);

    public Vector Forward(Vector input, ForwardContext context)
    {
        var z = weights * input + biases;
        var a = TActivation.Apply(z);
        context.Inputs[this] = input;
        context.PreActivations[this] = z;
        context.Activations[this] = a;
        return a;
    }

    public Vector Backward(Vector gradient, ForwardContext context, double learningRate)
    {
        var z = context.PreActivations[this];
        var aGrad = TActivation.Derivative(z);
        var delta = aGrad.PointwiseMultiply(gradient);

        var input = context.Inputs[this];
        var weightGrad = delta.OuterProduct(input);

        weights -= learningRate * weightGrad;
        biases -= learningRate * delta;

        return weights.TransposeThisAndMultiply(delta);
    }

    public void Initialize()
    {
        weights = Matrix.Build.Random(OutputSize, InputSize);
        biases.Clear();
    }

    public void Load(BinaryReader reader)
    {
        for (var i = 0; i < OutputSize; i++)
        {
            for (var j = 0; j < InputSize; j++)
            {
                weights[i, j] = reader.ReadDouble();
            }
        }
        for (var i = 0; i < OutputSize; i++)
        {
            biases[i] = reader.ReadDouble();
        }
    }

    public void Save(BinaryWriter writer)
    {
        for (var i = 0; i < OutputSize; i++)
        {
            for (var j = 0; j < InputSize; j++)
            {
                writer.Write(weights[i, j]);
            }
        }
        for (var i = 0; i < OutputSize; i++)
        {
            writer.Write(biases[i]);
        }
    }
}