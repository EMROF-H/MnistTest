global using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
global using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
global using Network = NeuralNetwork.NeuralNetwork;

using MnistTest.Mnist;
using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.Layers;
using ShellProgressBar;

static class Program
{
    internal static void Main()
    {
        // 0. Read Dataset
        var dataset = new CsvDataset(CsvDatasetPath.Train);

        // 1. Initialize Network
        var net = Initialize();

        // 2. Train
        Train(net, dataset, 10);

        // 3. Test
        Test(net);

        // 4. Save
        using var fs = File.Create(CsvDatasetPath.TrainedModel);
        net.SaveParameters(fs);
    }

    static Network Initialize()
    {
        var net = new Network();

        // 输入 28×28 = 784，输出 10（分类结果）
        // 隐层 128 个神经元，使用 ReLU
        net.AddLayer(new DenseLayer<IActivationFunction.Relu>(784, 128));

        // 输出层使用 Softmax 激活（这里我们先用 Sigmoid 模拟，Softmax 可以留作未来 vector 实现）
        net.AddLayer(new DenseLayer<IActivationFunction.Sigmoid>(128, 10));

        // 初始化参数
        net.Initialize();

        return net;
    }

    // 归一化函数
    static Vector NormalizePixels(byte[,] pixels)
    {
        var vec = Vector.Build.Dense(CsvSample.PixelCount);
        for (int y = 0; y < CsvSample.Height; y++)
        for (int x = 0; x < CsvSample.Width; x++)
            vec[y * CsvSample.Width + x] = pixels[y, x] / 255.0;
        return vec;
    }

    // One-hot 编码函数
    static Vector OneHot(byte label)
    {
        var v = Vector.Build.Dense(10);
        v[label] = 1.0;
        return v;
    }

    static void Train(Network net, CsvDataset dataset, int epochs)
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
            Train(net, dataset); // 调用单轮训练
            epochBar.Tick($"Epoch {epoch}/{epochs}");
        }
    }

    static void Train(Network net, CsvDataset dataset)
    {
        var rng = new Random();
        var samples = dataset
            .Samples
            .OrderBy(_ => rng.Next())
            .ToArray();

        var options = new ProgressBarOptions
        {
            ForegroundColor = ConsoleColor.Yellow,
            BackgroundColor = ConsoleColor.DarkGray,
            ProgressCharacter = '─'
        };

        using var progress = new ProgressBar(samples.Length, "Training MNIST...", options);
        for (int i = 0; i < samples.Length; i++)
        {
            var sample = samples[i];
            var input = NormalizePixels(sample.Pixels);
            var expected = OneHot(sample.Label);
            net.Train(input, expected, learningRate: 0.01);

            progress.Tick($"Training {i + 1}/{samples.Length}");
        }
    }


    static void Test(Network net)
    {
        var dataset = new CsvDataset(CsvDatasetPath.Test);
        var samples = dataset.Samples;

        var correct = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            var sample = samples[i];
            var input = NormalizePixels(sample.Pixels);
            var output = net.Predict(input);
            var predicted = output.MaximumIndex();
            int actual = sample.Label;

            var match = predicted == actual;
            if (match) correct++;

            Console.WriteLine($"[{i + 1}/{samples.Count}] Actual: {actual}, Predicted: {predicted}, {(match ? "Correct" : "Incorrect")}");
        }

        double accuracy = correct * 100.0 / samples.Count;
        Console.WriteLine($"\nTotal: {samples.Count}, Correct: {correct}, Accuracy: {accuracy:F2}%");
    }
}
