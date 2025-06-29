# 🧠 MnistTest - 教学用 C# 神经网络项目

本项目旨在提供一个结构清晰、可定制、可扩展的神经网络框架，适合用于学习与实践神经网络的基本原理。

项目使用 C# 编写，支持构建任意结构的神经网络，并通过标准手写数字识别数据集 [MNIST](http://yann.lecun.com/exdb/mnist/) 进行训练与评估。

---

## ✨ 项目特点

- 使用纯 C# 构建，无需依赖外部深度学习框架；
- 支持自定义层（如 `DenseLayer`）、激活函数（如 `ReLU`, `Sigmoid`, `Tanh`）；
- 拥有完整的训练、推理、保存与加载流程；
- 使用 `MathNet.Numerics` 作为底层矩阵计算库；
- 支持命令行进度条展示训练过程；
- 结构清晰，便于理解神经网络从零开始的搭建过程。

---

## ⚠️ 数据集说明

使用本项目前，请准备 MNIST CSV 格式数据文件。你可以使用已转换好的 CSV 数据集，例如：

- `mnist_train.csv`
- `mnist_test.csv`

并将其放置在指定目录下。项目中使用如下路径（定义于 `CsvDatasetPath.cs`）：

```csharp
public static class CsvDatasetPath 
{
    public static string Directory => @"C:\Users\EMROF\Documents\MnistTest\MNIST_CSV";
    
    public static string Train => @$"{Directory}\mnist_train.csv";
    public static string Test => @$"{Directory}\mnist_test.csv";
    
    public static string TrainedModel => @$"{Directory}\trained_model.bin";
}
```

> ✅ **请根据你的实际文件路径修改该类中的 `Directory` 字段。**

---

## 🧩 项目结构

```
MnistTest/
├── Benchmark/
├── Mnist/
│   ├── CsvDatasetPath.cs
│   ├── CsvSample.cs
│   ├── Dataset.cs
│   ├── MnistNetwork.cs
│   └── Program.cs
├── NeuralNetwork/
│   ├── ActivationFunctions/
│   │   ├── IActivationFunction.cs
│   │   ├── Identity.cs
│   │   ├── Relu.cs
│   │   ├── Sigmoid.cs
│   │   └── Tanh.cs
│   ├── Layers/
│   │   ├── DenseLayer.cs
│   │   ├── ForwardContext.cs
│   │   └── ILayer.cs
│   ├── GlobalUsingDouble.cs
│   └── NeuralNetwork.cs
```

---

## 🚀 快速开始

1. 克隆项目：

```bash
git clone https://github.com/your-name/MnistTest.git
```

2. 下载并准备 MNIST CSV 文件，并将其放入你设定的目录。

3. 编译并运行：

```bash
dotnet run --project MnistTest
```

---

## 🛠️ 示例代码片段

```csharp
var net = new NeuralNetwork();
net.AddLayer(new DenseLayer<Relu>(784, 64));
net.AddLayer(new DenseLayer<Sigmoid>(64, 10));
net.Initialize();

net.Train(input, expected, learningRate: 0.01);
var result = net.Predict(input);
```

---

## 📚 适合人群

- 想要了解神经网络底层原理的开发者；
- 想在 C# 中构建深度学习框架的学习者；
- 希望进行小规模实验或教学演示的老师和学生。

---

## 📝 TODO

- [ ] 添加卷积层（ConvolutionLayer）
- [ ] 支持批量训练（Batch Training）
- [ ] 增加 Dropout、正则化等机制
- [ ] 实现网络结构图可视化

---

## 📄 License

MIT License.
