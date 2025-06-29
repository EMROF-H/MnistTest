namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public class Identity : IActivationFunction
    {
        static double IActivationFunction.Apply(double x) => x;
    }
}
