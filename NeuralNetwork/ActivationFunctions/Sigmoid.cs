namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public sealed class Sigmoid : IScalar<Sigmoid>
    {
        private Sigmoid() { }

        public static double Apply(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double Derivative(double x)
        {
            var y = 1.0 / (1.0 + Math.Exp(-x));
            return y * (1 - y);
        }
    }
}
