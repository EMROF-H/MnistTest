using MnistTest.Mnist;

var dataset = new CsvDataset(CsvDatasetPath.Train);

foreach (var sample in dataset.Samples)
{
    Console.WriteLine($"Label: {sample.Label}, First pixel: {sample.Pixels[0, 0]}");
}
