global using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
global using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
global using Network = NeuralNetwork.NeuralNetwork;

using MnistTest.Mnist;
using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.Layers;
using ShellProgressBar;

namespace MnistTest;

class MnistNetwork
{
    public Network Net { get; } = new();

    public MnistNetwork()
    {
        Build();
        Net.Initialize();
    }
    
    public MnistNetwork(string path)
    {
        Build();
        Load(path);
    }
    
    private void Build()
    {
        // 输入 28×28 = 784，输出 10（分类结果）
        // 隐层 128 个神经元，使用 ReLU
        Net.AddLayer(new DenseLayer<IActivationFunction.Relu>(784, 128));

        // 输出层使用 Softmax 激活（这里我们先用 Sigmoid 模拟，Softmax 可以留作未来 vector 实现）
        Net.AddLayer(new DenseLayer<IActivationFunction.Sigmoid>(128, 10));

        // // 输入层 → 隐藏层1
        // Net.AddLayer(new DenseLayer<IActivationFunction.Relu>(784, 512));
        // Net.AddLayer(new DropoutLayer(0.2));
        //
        // // 隐藏层2
        // Net.AddLayer(new DenseLayer<IActivationFunction.Relu>(512, 256));
        // Net.AddLayer(new DropoutLayer(0.2));
        //
        // // 隐藏层3
        // Net.AddLayer(new DenseLayer<IActivationFunction.Relu>(256, 128));
        // Net.AddLayer(new DropoutLayer(0.2));
        //
        // // 输出层使用 Softmax（建议实现 vector 版本）
        // Net.AddLayer(new DenseLayer<IActivationFunction.Sigmoid>(128, 10));
    }

    private static Vector NormalizePixels(byte[,] pixels)
    {
        var vec = Vector.Build.Dense(CsvSample.PixelCount);
        for (var y = 0; y < CsvSample.Height; y++)
        for (var x = 0; x < CsvSample.Width; x++)
            vec[y * CsvSample.Width + x] = pixels[y, x] / 255.0;
        return vec;
    }

    private static Vector OneHot(byte label)
    {
        var v = Vector.Build.Dense(10);
        v[label] = 1.0;
        return v;
    }

    public void Train(CsvDataset dataset, int epochs)
    {
        var outerOptions = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Cyan,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─'
        };

        using var epochBar = new ProgressBar(epochs, "Training epochs...", outerOptions);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            TrainOnce(dataset, epoch); // 调用单轮训练
            epochBar.Tick($"Epoch {epoch}/{epochs}");
        }
    }

    public void TrainOnce(CsvDataset dataset, int? thisTime = null)
    {
        var random = new Random();
        var samples = dataset
            .Samples
            .OrderBy(_ => random.Next())
            .ToArray();

        var options = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Yellow,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─'
        };

        using var progress = new ProgressBar(samples.Length, "Training MNIST ...", options);
        for (var i = 0; i < samples.Length; i++)
        {
            var sample = samples[i];
            var input = NormalizePixels(sample.Pixels);
            var expected = OneHot(sample.Label);
            Net.Train(input, expected, learningRate: 0.01);

            progress.Tick($"Training{(thisTime is null ? "" : $" {thisTime.Value.ToString()} times")} {i + 1}/{samples.Length}");
        }
    }
    
    public double Test(CsvDataset dataset)
    {
        var samples = dataset.Samples;

        var correct = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            var sample = samples[i];
            var input = NormalizePixels(sample.Pixels);
            var output = Net.Predict(input);
            var predicted = output.MaximumIndex();
            int actual = sample.Label;

            var match = predicted == actual;
            if (match) correct++;

            // Console.WriteLine($"[{i + 1}/{samples.Count}] Actual: {actual}, Predicted: {predicted}, {(match ? "Correct" : "Incorrect")}");
        }

        var accuracy = correct * 100.0 / samples.Count;
        // Console.WriteLine();
        // Console.WriteLine($"Total: {samples.Count}, Correct: {correct}, Accuracy: {accuracy:F2}%");
        return accuracy;
    }

    public void Load(string path)
    {
        using var f = File.OpenRead(path);
        Net.LoadParameters(f);
    }
    
    public void Save(string path)
    {
        using var fs = File.Create(path);
        Net.SaveParameters(fs);
    }
}
