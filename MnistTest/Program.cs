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

// var net = new MnistNetwork();
// net.Save($@"{CsvDatasetPath.Directory}\0_{DateTime.Now:yy-MM-dd-HH-mm-ss-fff}.bin");
//
// var trainSet = new CsvDataset(CsvDatasetPath.Train);
// var testSet = new CsvDataset(CsvDatasetPath.Test);
//
// for (var i = 1; i <= 100; i++)
// {
//     net.TrainOnce(trainSet, i);
//     var accuracy = net.Test(testSet);
//     net.Save($@"{CsvDatasetPath.Directory}\{i}_{DateTime.Now:yy-MM-dd-HH-mm-ss-fff}_{accuracy:F2}.bin");
// }
