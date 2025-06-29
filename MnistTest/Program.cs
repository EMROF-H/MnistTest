using MnistTest;
using MnistTest.Mnist;

var net = new MnistNetwork();
// var net = new MnistNetwork($@"{CsvDatasetPath.Directory}\trained_model_10_2025062901_0.4116.bin");

var trainSet = new CsvDataset(CsvDatasetPath.Train);
net.Train(trainSet, 10);

var testSet = new CsvDataset(CsvDatasetPath.Test);
net.Test(testSet);

net.Save(CsvDatasetPath.TrainedModel);

Console.Beep();
