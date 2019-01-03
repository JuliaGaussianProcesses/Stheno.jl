using Revise
using Stheno, BenchmarkTools, Bench, Flux, Zygote
using Zygote: gradient
using BenchmarkTools: median

# Various bits of utility for benchmarking.
include("util.jl")

println("Constructing")
Bench.clear_suites()
benchmarks = @benchset "Stheno" begin

    # @benchset "mean_and_kernel" begin
    #     include("mean_and_kernel/mean.jl")
        include("mean_and_kernel/kernel.jl")
        # include("mean_and_kernel/compose.jl")
        # include("mean_and_kernel/finite.jl")
    # end

    # @benchset "gp" begin
    #     include("gp/abstract_gp.jl")
    # end
end

println("Tuning")
tune!(benchmarks; verbose=true)
println("Running")
results = run(benchmarks; verbose=true)

println("Time:")
pretty_print(median(results))

println()
println("Memory:")
pretty_print(memory(results))

println()
println("Allocs:")
pretty_print(allocs(results))
