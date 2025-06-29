namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public class Tanh : IActivationFunction
    {
        static double IActivationFunction.Apply(double x) => Math.Tanh(x);
        static double IActivationFunction.Derivative(double x)
        {
            var y = Math.Tanh(x);
            return 1 - y * y;
        }
    }
}
