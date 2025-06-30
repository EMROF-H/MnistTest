namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public sealed class Relu : IScalar<Relu>
    {
        private Relu() { }

        public static double Apply(double x) => Math.Max(0, x);
        public static double Derivative(double x) => x > 0 ? 1.0 : 0.0;
    }
}
