function mean_benchmarks()
    suite = BenchmarkGroup()
    # suite["ZeroMean{Float64}()"] = create_benchmarks(ZeroMean{Float64}())
    # suite["ConstantMean(5.0)"] = create_benchmarks(ConstantMean(5.0))
    # suite["CustomMean(sin)"] = create_benchmarks(CustomMean(sin))
    # suite["CustomMean(cos)"] = create_benchmarks(CustomMean(cos))
    # suite["sin + cos"] = create_benchmarks(CustomMean(sin) + CustomMean(cos))
    suite["5 * CustomMean(sin)"] = create_benchmarks(5.0 * CustomMean(sin))
    return suite
end

function create_benchmarks(μ::MeanFunction)
    suite = BenchmarkGroup()

    suite["μ(x)"] = @benchmarkable $μ(5.0)
    suite["gradient(μ, x)"] = @benchmarkable Zygote.gradient($μ, 5.0)

    for N in [10, 100, 100, 1_000]
        x = randn(N)
        suite["map ($N)"] = @benchmarkable map($μ, $x)
        suite["gradient map ($N)"] = @benchmarkable Zygote.gradient(x->sum(map($μ, x)), $x)
    end

    return suite
end
