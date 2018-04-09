using Revise
using Stheno, Test, BenchmarkTools, QuadGK, Random, LinearAlgebra, BlockArrays

const check_mem = false

@testset "Stheno" begin

    include("block_arrays.jl")
    include("covariance_matrices.jl")

    @testset "mean_and_kernel" begin
        include("mean_and_kernel/mean.jl")
        include("mean_and_kernel/kernel.jl")
        # include("mean_and_kernel/compose.jl ")
        include("mean_and_kernel/conditional.jl")
        include("mean_and_kernel/finite.jl")
        include("mean_and_kernel/cat.jl")
        # include("mean_and_kernel/transform.jl")
        # include("mean_and_kernel/input_transform.jl")
    end

    include("gp.jl")

    @testset "linops" begin
        include("lin_ops.jl")
    #     include("linops/addition.jl")
    #     include("linops/product.jl")
    #     include("linops/integrate.jl")
    end
end
