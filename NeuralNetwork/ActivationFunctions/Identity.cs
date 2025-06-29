namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public class Identity : IActivationFunction
    {
        static double IActivationFunction.Apply(double x) => x;
        static double IActivationFunction.Derivative(double x) => 1.0;
    }
}
