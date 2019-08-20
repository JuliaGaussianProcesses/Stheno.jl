using Stheno, Test, Random, BlockArrays, StatsFuns
using BlockArrays: _BlockArray

using Stheno: ew, pw, mean_vector, cov, cov_diag
using Stheno: EQ

include("test_util.jl")

@testset "Stheno" begin

    # println("testing util")
    # a = time()
    # @testset "util" begin
    #     include(joinpath("util", "zygote_rules.jl"))
    #     include(joinpath("util", "covariance_matrices.jl"))
    #     @testset "block_arrays" begin
    #         include(joinpath("util", "block_arrays", "test_util.jl"))
    #         include(joinpath("util", "block_arrays", "dense.jl"))
    #         include(joinpath("util", "block_arrays", "diagonal.jl"))
    #     end
    #     include(joinpath("util", "abstract_data_set.jl"))
    # end
    # println("testing util took $(time() - a)s")

    # println("testing mean_and_kernel")
    # a = time()
    # @testset "mean_and_kernel" begin
    #     include(joinpath("mean_and_kernel", "mean.jl"))
    #     include(joinpath("mean_and_kernel", "kernel.jl"))
    # end
    # println("mean_and_kernel took $(time() - a)s")


    # println("testing gp")
    # a = time()
    # @testset "gp" begin
    #     include(joinpath("gp", "gp.jl"))
    #     include(joinpath("gp", "finite_gp.jl"))
    # end
    # println("testing gp took $(time() - a)s")

    println("linops:")
    @time @testset "linops" begin
        include(joinpath("linops", "test_util.jl"))
        include(joinpath("linops", "indexing.jl"))
        include(joinpath("linops", "cross.jl"))
        include(joinpath("linops", "conditioning.jl"))
        include(joinpath("linops", "product.jl"))
        include(joinpath("linops", "addition.jl"))
        include(joinpath("linops", "compose.jl"))
        # include(joinpath("linops", "approximate_conditioning.jl"))
    end
end
