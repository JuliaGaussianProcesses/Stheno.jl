using Stheno, Test, Random, FillArrays, BlockArrays
using BlockArrays: _BlockArray

include("test_util.jl")

@testset "Stheno" begin

    # @testset "util" begin
    #     include("util/zygote_rules.jl")
        # include("util/covariance_matrices.jl")
    #     @testset "block_arrays" begin
            include("util/block_arrays/test_util.jl")
    #         include("util/block_arrays/dense.jl")
            include("util/block_arrays/diagonal.jl")
    #         # include("util/block_arrays/triangular.jl")
    #     end
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
        # include("gp/finite_gp.jl") # run these later
    end

    # @testset "linops" begin
    #     include("linops/indexing.jl")
    #     include("linops/conditioning.jl")
    #     include("linops/product.jl")
    #     include("linops/addition.jl")
    #     include("linops/compose.jl")
    #     include("linops/approximate_conditioning.jl")
    #     include("linops/integrate.jl")
    # end

    @testset "integration" begin
        # include("util/toeplitz_integration.jl")
    end
end
