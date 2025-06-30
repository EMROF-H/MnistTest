namespace NeuralNetwork.ActivationFunctions;

public partial interface IActivationFunction
{
    public sealed class Identity : IScalar<Identity>
    {
        private Identity() { }

        public static double Apply(double x) => x;
        public static double Derivative(double x) => 1.0;
    }
}
