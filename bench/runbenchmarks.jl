using Revise
using Stheno, BenchmarkTools, Flux, Zygote
using BenchmarkTools: median
# plotly();

# What settings should we require on the user in terms of threading? Should we require any?
# Should we just log the environment variables and display them with the summary?


# #####################################
# # High-level integration benchmarks #
# #####################################

# function dense_1D_model(m, k, N::Int)
#     f = GP(m, k, GPC())
#     x = randn(N)
#     y = rand(f(x))
#     return logpdf(f(x), y)
# end

# # Collection of small-medium sized problems.
# dense_1D = BenchmarkGroup()
# Ns = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
# for N in Ns
#     dense_1D[N] = @benchmarkable dense_1D_model(0, EQ(), $N)
# end

# tune!(dense_1D)
# results = run(dense_1D, verbose=true, seconds=1)

# times = time(median(dense_1D))

# kys, vals = Vector{Any}(undef, length(Ns)), Vector{Float64}(undef, length(Ns))
# for (p, N) in enumerate(Ns)
#     kys[p] = p
#     vals[p] = times[N]
# end

# plot(kys, vals ./ 1e9; xscale=:log10, yscale=:log10)


# function get_times()

include("mean_and_kernel/mean.jl")
include("mean_and_kernel/kernel.jl")

function construct_benchmarks()
    suite = BenchmarkGroup()
    # suite["mean_and_kernel/mean.jl"] = mean_benchmarks()
    suite["mean_and_kernel/kernel.jl"] = kernel_benchmarks()
    return suite
end

println("Constructing:")
suite = construct_benchmarks()
println("Tuning:")
tune!(suite)
println("Running:")
results = run(suite, verbose=true)


function pretty_print(d::BenchmarkGroup, pre=1)
    pretty_print(d.data, pre)
end

function pretty_print(d::Dict, pre=1)
    for (k, v) in d
        if v isa BenchmarkGroup
            s = "$(repr(k)) => "
            println(join(fill(" ", pre)) * s)
            pretty_print(v, pre+1+4)
        else
            println(join(fill(" ", pre)) * "$(repr(k)) => $(repr(v))")
        end
    end
    nothing
end

println("Time:")
pretty_print(median(results))

println()
println("Memory:")
pretty_print(memory(results))

println()
println("Allocs:")
pretty_print(allocs(results))



