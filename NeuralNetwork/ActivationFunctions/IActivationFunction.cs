namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    static abstract Vector Apply(Vector vector);
    static abstract Vector Derivative(Vector vector);
}

public interface ISingleActivationFunction<TSelf> : IActivationFunction where TSelf : ISingleActivationFunction<TSelf>
{
    static abstract double Apply(double x);
    static abstract double Derivative(double x);

    static Vector IActivationFunction.Apply(Vector v) => v.Map(static x => TSelf.Apply(x));
    static Vector IActivationFunction.Derivative(Vector v) => v.Map(static x => TSelf.Derivative(x));
}
