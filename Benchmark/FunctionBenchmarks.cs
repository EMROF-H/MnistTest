using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;

namespace Benchmark;

// ① 静态方法（普通）
public static class StaticMethod
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
}

// ② 接口 + 实例方法
public interface ISigmoid
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    double Sigmoid(double x);
}

public class InterfaceMethod : ISigmoid
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
}

// ③ 接口属性返回静态方法委托
public interface IDelegateProperty
{
    Func<double, double> SigmoidFunc { get; }
}

public class DelegatePropertyImpl : IDelegateProperty
{
    public Func<double, double> SigmoidFunc => StaticMethod.Sigmoid;
}

// ④ 实例字段中的 lambda 委托
public class StoredDelegate
{
    private readonly Func<double, double> _sigmoid = x => 1 / (1 + Math.Exp(-x));
    [MethodImpl(MethodImplOptions.NoInlining)]
    public double Invoke(double x) => _sigmoid(x);
}

// ⑤ 接口属性返回 lambda 委托
public interface ILambdaProvider
{
    Func<double, double> Lambda { get; }
}

public class LambdaImpl : ILambdaProvider
{
    public Func<double, double> Lambda => x => 1 / (1 + Math.Exp(-x));
}

// ⑥ C# 11+ 静态接口方法
public interface ISigmoidStatic<TSelf> where TSelf : ISigmoidStatic<TSelf>
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    static abstract double Sigmoid(double x);
}

public class StaticVirtualImpl : ISigmoidStatic<StaticVirtualImpl>
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
}

// 泛型调度器：调用静态接口成员
public static class GenericCall<T> where T : ISigmoidStatic<T>
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static double Sigmoid(double x) => T.Sigmoid(x);
}

[MemoryDiagnoser, RPlotExporter]
public class FunctionBenchmarks
{
    private const double Input = 0.75;
    private const int Times = 10000;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    [Benchmark]
    public void CommonMethodCall()
    {
        for (var i = 0; i < Times; i++)
        {
            Sigmoid(Input);
        }
    }

    [Benchmark]
    public void StaticMethodCall()
    {
        for (var i = 0; i < Times; i++)
        {
            StaticMethod.Sigmoid(Input);
        }
    }
    
    private readonly ISigmoid _interface = new InterfaceMethod();
    [Benchmark]
    public void InterfaceMethodCall()
    {
        for (var i = 0; i < Times; i++)
        {
            _interface.Sigmoid(Input);
        }
    }

    private readonly IDelegateProperty _delegateProp = new DelegatePropertyImpl();
    [Benchmark]
    public void DelegatePropertyCall()
    {
        for (var i = 0; i < Times; i++)
        {
            _delegateProp.SigmoidFunc(Input);
        }
    }

    private readonly StoredDelegate _storedDelegate = new StoredDelegate();
    [Benchmark]
    public void StoredDelegateCall()
    {
        for (var i = 0; i < Times; i++)
        {
            _storedDelegate.Invoke(Input);
        }
    }

    private readonly ILambdaProvider _lambda = new LambdaImpl();
    [Benchmark]
    public void LambdaPropertyCall()
    {
        for (var i = 0; i < Times; i++)
        {
            _lambda.Lambda(Input);
        }
    }

    [Benchmark]
    public void StaticVirtualCall()
    {
        for (var i = 0; i < Times; i++)
        {
            GenericCall<StaticVirtualImpl>.Sigmoid(Input);
        }
    } 
}
