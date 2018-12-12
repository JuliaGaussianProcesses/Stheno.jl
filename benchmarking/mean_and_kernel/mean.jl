function mean_benchmarks()
    suite = BenchmarkGroup()
    # suite["ZeroMean{Float64}()"] = create_benchmarks(ZeroMean{Float64}())
    # suite["ConstantMean(5.0)"] = create_benchmarks(ConstantMean(5.0))
    suite["CustomMean(sin)"] = create_benchmarks(CustomMean(sin))
    suite["CustomMean(cos)"] = create_benchmarks(CustomMean(cos))
    return suite
end

const mean_Ns = [10, 100, 100, 10_000]

function create_benchmarks(μ::MeanFunction)
    suite = BenchmarkGroup()

    suite["μ(x)"] = @benchmarkable $μ(5.0)
    suite["gradient(μ, x)"] = @benchmarkable Zygote.gradient($μ, 5.0)

    for N in mean_Ns
        x = randn(N)
        suite["map ($N)"] = @benchmarkable map($μ, $x)
        suite["gradient map ($N)"] = @benchmarkable Zygote.gradient(x->sum(map($μ, x)), $x)
    end

    return suite
end
