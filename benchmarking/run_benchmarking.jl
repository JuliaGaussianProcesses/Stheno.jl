using Stheno, BenchmarkTools, Plots
using BenchmarkTools: median
plotly();

# What settings should we require on the user in terms of threading? Should we require any?
# Should we just log the environment variables and display them with the summary?


#####################################
# High-level integration benchmarks #
#####################################

function dense_1D_model(m, k, N::Int)
    f = GP(m, k, GPC())
    x = randn(N)
    y = rand(f(x))
    return logpdf(f(x), y)
end

# Collection of small-medium sized problems.
dense_1D = BenchmarkGroup()
Ns = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
for N in Ns
    dense_1D[N] = @benchmarkable dense_1D_model(0, EQ(), $N)
end

tune!(dense_1D)
results = run(dense_1D, verbose=true, seconds=1)

times = time(median(dense_1D))

kys, vals = Vector{Any}(undef, length(Ns)), Vector{Float64}(undef, length(Ns))
for (p, N) in enumerate(Ns)
    kys[p] = p
    vals[p] = times[N]
end

plot(kys, vals ./ 1e9; xscale=:log10, yscale=:log10)


function get_times()
