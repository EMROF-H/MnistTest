namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public class Sigmoid : IActivationFunction
    {
        static double IActivationFunction.Apply(double x) => 1.0 / (1.0 + Math.Exp(-x));
    }
}
