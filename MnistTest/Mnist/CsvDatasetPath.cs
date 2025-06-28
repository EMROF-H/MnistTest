namespace MnistTest.Mnist;

public static class CsvDatasetPath
{
    public static string Directory => @"C:\Users\EMROF\Documents\MnistTest\MNIST_CSV";
    
    public static string Train => @$"{Directory}\mnist_train.csv";
    public static string Test => @$"{Directory}\mnist_test.csv";
}