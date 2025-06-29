namespace NeuralNetwork.Neurons;

public class ForwardContext;

public abstract class Neuron
{
    public abstract double Output { get; }
    public abstract void Forward(ForwardContext context);
}
