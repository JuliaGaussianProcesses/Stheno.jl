function kernel_benchmarks()
    suite = BenchmarkGroup()
    suite["ZeroKernel{Float64}()"] = create_benchmarks(ZeroKernel{Float64}())
    suite["ConstantKernel(5.0)"] = create_benchmarks(ConstantKernel(5.0))
    suite["EQ Real"] = create_benchmarks(EQ())
    return suite
end

function create_benchmarks(k::Kernel, x=5.0, x′=4.0)
    suite = BenchmarkGroup()

    suite["k(x)"] = @benchmarkable $k($x)
    suite["k(x, x′)"] = @benchmarkable $k($x, $x′)

    for N in [10, 100, 1_000, 10_000]
        x̄, x̄′ = fill(x, N), fill(x′, N)
        suite["map k(x̄) $N"] = @benchmarkable map($k, $x̄)
        suite["map k(x̄, x̄′) $N"] = @benchmarkable map($k, $x̄, $x̄′)
        suite["pairwise k(x̄) $N"] = @benchmarkable pairwise($k, $x̄)
        suite["pairwise k(x, x̄′) $N"] = @benchmarkable pairwise($k, $x̄, $x̄′)
    end

    return suite
end


# function mean_benchmarks()
#     suite = BenchmarkGroup()
#     # suite["ZeroMean{Float64}()"] = create_benchmarks(ZeroMean{Float64}())
#     # suite["ConstantMean(5.0)"] = create_benchmarks(ConstantMean(5.0))
#     # suite["CustomMean(sin)"] = create_benchmarks(CustomMean(sin))
#     # suite["CustomMean(cos)"] = create_benchmarks(CustomMean(cos))
#     # suite["sin + cos"] = create_benchmarks(CustomMean(sin) + CustomMean(cos))
#     suite["5 * CustomMean(sin)"] = create_benchmarks(5.0 * CustomMean(sin))
#     return suite
# end

# function create_benchmarks(μ::MeanFunction)
#     suite = BenchmarkGroup()

#     suite["μ(x)"] = @benchmarkable $μ(5.0)
#     suite["gradient(μ, x)"] = @benchmarkable Zygote.gradient($μ, 5.0)

#     for N in [10, 100, 1000, 10_000]
#         x = randn(N)
#         suite["map ($N)"] = @benchmarkable map($μ, $x)
#         suite["gradient map ($N)"] = @benchmarkable Zygote.gradient(x->sum(map($μ, x)), $x)
#     end

#     return suite
# end
