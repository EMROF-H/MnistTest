namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public sealed class Tanh : IScalar<Tanh>
    {
        private Tanh() { }

        public static double Apply(double x) => Math.Tanh(x);
        public static double Derivative(double x)
        {
            var y = Math.Tanh(x);
            return 1 - y * y;
        }
    }
}
