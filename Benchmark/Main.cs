using Benchmark;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Running;

var r = BenchmarkRunner.Run<FunctionBenchmarks>();
RPlotExporter.Default.ExportToFiles(r, new StreamLogger(@"C:\Users\EMROF\Documents\MnistTest\benchmark"));
