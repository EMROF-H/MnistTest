namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    static abstract Vector Apply(Vector vector);
    static abstract Vector Derivative(Vector vector);
}
