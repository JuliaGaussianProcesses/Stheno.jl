using Stheno, Test, Random, BlockArrays, StatsFuns
using BlockArrays: _BlockArray

include("test_util.jl")

@testset "Stheno" begin

    @testset "util" begin
        include(joinpath("util", "zygote_rules.jl"))
        include(joinpath("util", "covariance_matrices.jl"))
        @testset "block_arrays" begin
            include(joinpath("util", "block_arrays", "test_util.jl"))
            include(joinpath("util", "block_arrays", "dense.jl"))
            include(joinpath("util", "block_arrays", "diagonal.jl"))
        end
        include(joinpath("util", "abstract_data_set.jl"))
    end

    @testset "mean_and_kernel" begin
        include(joinpath("mean_and_kernel", "mean.jl"))
        include(joinpath("mean_and_kernel", "kernel.jl"))
        # include(joinpath("mean_and_kernel", "compose.jl"))
        # include(joinpath("mean_and_kernel", "block.jl"))
        # include(joinpath("mean_and_kernel", "input_transform.jl"))
        # # include(joinpath("mean_and_kernel", "derivative.jl")) # These tests currenly fail because Zygote.
        # @testset "conditioning" begin
        #     include(joinpath("mean_and_kernel", "conditioning", "exact.jl"))
        #     include(joinpath("mean_and_kernel", "conditioning", "titsias.jl"))
        # end
        # include(joinpath("mean_and_kernel", "algebra.jl"))
        # include(joinpath("mean_and_kernel", "util.jl"))
    end

    @testset "gp" begin
        include(joinpath("gp", "gp.jl"))
        include(joinpath("gp", "finite_gp.jl"))
    end

    # @testset "linops" begin
    #     include(joinpath("linops", "test_util.jl"))
    #     include(joinpath("linops", "indexing.jl"))
    #     include(joinpath("linops", "cross.jl"))
    #     include(joinpath("linops", "conditioning.jl"))
    #     include(joinpath("linops", "product.jl"))
    #     include(joinpath("linops", "addition.jl"))
    #     include(joinpath("linops", "compose.jl"))
    #     # include(joinpath("linops", "approximate_conditioning.jl"))
    #     # include(joinpath("linops", "integrate.jl"))
    # end
end
