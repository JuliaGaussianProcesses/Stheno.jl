using Stheno, Test, Random, FillArrays

include("test_util.jl")

@testset "Stheno" begin

    # @testset "util" begin
    #     include("util/zygote_rules.jl")
    #     include("util/covariance_matrices.jl")
        @testset "block_arrays" begin
            include("util/block_arrays/test_util.jl")
            include("util/block_arrays/dense.jl")
            include("util/block_arrays/diagonal.jl")
            include("util/block_arrays/triangular.jl")
        end
    #     include("util/abstract_data_set.jl")
    #     include("util/fillarrays.jl")
    # end

    # @testset "mean_and_kernel" begin
    #     include("mean_and_kernel/mean.jl")
    #     include("mean_and_kernel/kernel.jl")
    #     include("mean_and_kernel/compose.jl")
    #     include("mean_and_kernel/block.jl")
    #     include("mean_and_kernel/input_transform.jl")
    #     # include("mean_and_kernel/derivative.jl") # These tests currenly fail because Zygote.
    #     @testset "conditioning" begin
    #         include("mean_and_kernel/conditioning/exact.jl")
    #         include("mean_and_kernel/conditioning/titsias.jl")
    #     end
    #     include("mean_and_kernel/algebra.jl")
    #     include("mean_and_kernel/util.jl")
    # end

    @testset "gp" begin
        # include("gp/gp.jl")
        # include("gp/block_gp.jl")
        include("gp/finite_gp.jl")
    end

    @testset "linops" begin
        # include("linops/indexing.jl")
        # include("linops/conditioning.jl")
        # include("linops/product.jl")
        # include("linops/addition.jl")
        # include("linops/compose.jl")
        # include("linops/approximate_conditioning.jl")
        # include("linops/integrate.jl")
    end

    @testset "integration" begin
        # include("util/toeplitz_integration.jl")
    end
end

# using Stheno, Zygote
# using Stheno: pw

# D, N, N′ = 10, 1000, 1001
# X, X′ = ColsAreObs(randn(D, N)), ColsAreObs(randn(D, N′))

# # Standardised eq kernel with length-scale 0.1. 
# pw(eq(; l=0.1), X, X′)

# # Get the gradient w.r.t. the inputs.
# S, back = Zygote.forward((X, X′, l)->pw(eq(; l=l), X, X′), X, X′, 0.1)
# S̄ = randn(size(S))
# back(S̄)

# S, back = Zygote.forward((X, X′)->pw(eq(), X, X′), X, X′);
# S̄ = randn(size(S));
# back(S̄);



# using KernelFunctions
# using Stheno
# using Stheno: pw
# using BenchmarkTools
# using Zygote

# D = 500;
# A = randn(D,1000);
# B = randn(D,1001);

# # Standardised eq kernel with length-scale 0.1.
# @btime pw(eq(; l=0.1), ColsAreObs(A), ColsAreObs(B));
# @btime kernelmatrix(SquaredExponentialKernel(0.01),A,B,obsdim=2);




# using KernelFunctions
# using Stheno
# using Stheno: pw
# using BenchmarkTools
# using Zygote
# using ProgressMeter, Statistics

# Ds = [1,2,5,10,20,50,100,200,500,1000]
# # Ds = [1,10,100,1000]
# time_stheno = zeros(length(Ds))
# mem_stheno = zeros(length(Ds))
# allocs_stheno = zeros(length(Ds))
# time_kf = zeros(length(Ds))
# mem_kf = zeros(length(Ds))
# allocs_kf = zeros(length(Ds))
# @showprogress for (i,D) in enumerate(Ds)

#     A = randn(D,1000)
#     B = randn(D,1001)

#     # Benchmark Stheno
#     stheno_bench = median(@benchmark pw(eq(;l=1.0), ColsAreObs($A), ColsAreObs($B)))
#     time_stheno[i] = stheno_bench.time
#     mem_stheno[i] = stheno_bench.memory
#     allocs_stheno[i] = stheno_bench.allocs

#     # Benchmark KernelFunctions
#     kf_bench = median(@benchmark KernelFunctions.kernelmatrix(SquaredExponentialKernel(1.0), $A, $B, obsdim=2))
#     time_kf[i] = kf_bench.time
#     mem_kf[i] = kf_bench.memory
#     allocs_kf[i] = kf_bench.allocs
# end


# using Plots
# gr()

# plot(Ds, time_stheno ./ time_kf; label = "time", xaxis=:log, xlabel="D", ylabel = "stheno / kf")
# plot!(Ds, mem_stheno ./ mem_kf; label="memory")
# plot!(Ds, allocs_stheno ./ allocs_kf; label="allocs")
