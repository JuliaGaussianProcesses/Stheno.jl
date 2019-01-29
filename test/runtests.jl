using Stheno, Test, Random, FillArrays

include("test_util.jl")

@testset "Stheno" begin

    @testset "util" begin
        include("util/zygote_rules.jl")
        include("util/covariance_matrices.jl")
        include("util/block_arrays.jl")
        include("util/abstract_data_set.jl")
    end

    @testset "mean_and_kernel" begin
        include("mean_and_kernel/mean.jl")
        include("mean_and_kernel/kernel.jl")
        include("mean_and_kernel/compose.jl")
        # include("mean_and_kernel/block.jl") # These are currently out of action.
        include("mean_and_kernel/input_transform.jl")
        include("mean_and_kernel/degenerate.jl")
        # include("mean_and_kernel/derivative.jl") # These tests currenly fail because Zygote.
        include("mean_and_kernel/conditional.jl")
        include("mean_and_kernel/algebra.jl")
        include("mean_and_kernel/util.jl")
    end

    @testset "gp" begin
        include("gp/gp.jl")
    #     include("gp/block_gp.jl")
        include("gp/finite_gp.jl")
    end

    @testset "linops" begin
        include("linops/indexing.jl")
        include("linops/conditioning.jl")
        include("linops/product.jl")
        include("linops/addition.jl")
        include("linops/compose.jl")
        include("linops/project.jl")
        # include("linops/integrate.jl")
    end

    @testset "integration" begin
        # include("util/toeplitz_integration.jl")
    end
end
