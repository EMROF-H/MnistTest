namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public interface IScalar<TSelf> : IActivationFunction where TSelf : IScalar<TSelf>
    {
        static abstract double Apply(double x);
        static abstract double Derivative(double x);

        static Vector IActivationFunction.Apply(Vector v) => v.Map(static x => TSelf.Apply(x));
        static Vector IActivationFunction.Derivative(Vector v) => v.Map(static x => TSelf.Derivative(x));
    }
}
