using Stheno, Test, Random, LinearAlgebra, Statistics

@testset "Stheno" begin

    @testset "util" begin
        include("util/covariance_matrices.jl")
        include("util/woodbury.jl")
        include("util/block_arrays.jl")
        include("util/abstract_data_set.jl")
    end

    @testset "mean_and_kernel" begin
        include("test_util.jl")
        include("mean_and_kernel/mean.jl")
        include("mean_and_kernel/kernel.jl")
        include("mean_and_kernel/compose.jl")
        include("mean_and_kernel/finite.jl")
        include("mean_and_kernel/conditional.jl")
        include("mean_and_kernel/block.jl")
        # include("mean_and_kernel/transform.jl")
        include("mean_and_kernel/input_transform.jl")
        include("mean_and_kernel/zero.jl")
        include("mean_and_kernel/degenerate.jl")
        include("mean_and_kernel/derivative.jl")
        include("mean_and_kernel/conversion.jl")
    end

    @testset "gp" begin
        include("gp/abstract_gp.jl")
        include("gp/gp.jl")
        include("gp/block_gp.jl")
    end

    @testset "linops" begin
        include("linops/indexing.jl")
        include("linops/addition.jl")
        include("linops/product.jl")
        # include("linops/integrate.jl")
        include("linops/project.jl")
        include("linops/conditioning.jl")
    end

    @testset "integration" begin
        include("util/toeplitz_integration.jl")
    end
end
