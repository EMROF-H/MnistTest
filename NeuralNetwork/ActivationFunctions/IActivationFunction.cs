namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    static abstract double Apply(double x);
    static abstract double Derivative(double x);
}
