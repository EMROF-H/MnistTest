namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public class Relu : IActivationFunction
    {
        static double IActivationFunction.Apply(double x) => Math.Max(0, x);
        static double IActivationFunction.Derivative(double x) => x > 0 ? 1.0 : 0.0;
    }
}
