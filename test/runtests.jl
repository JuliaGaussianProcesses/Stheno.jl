using Stheno, Test, Random, BlockArrays, StatsFuns
using BlockArrays: _BlockArray

using Stheno: ew, pw, mean_vector, cov, cov_diag
using Stheno: EQ

include("test_util.jl")

@testset "Stheno" begin

    println("util:")
    @time @testset "util" begin
        include(joinpath("util", "zygote_rules.jl"))
        include(joinpath("util", "covariance_matrices.jl"))
        @testset "block_arrays" begin
            include(joinpath("util", "block_arrays", "test_util.jl"))
            include(joinpath("util", "block_arrays", "dense.jl"))
            include(joinpath("util", "block_arrays", "diagonal.jl"))
        end
        include(joinpath("util", "abstract_data_set.jl"))
    end

    println("abstract_gp:")
    @time include("abstract_gp.jl")

    println("gp:")
    @time @testset "gp" begin
        include(joinpath("gp", "mean.jl"))
        include(joinpath("gp", "kernel.jl"))
        include(joinpath("gp", "gp.jl"))
    end    

    println("composite:")
    @time @testset "composite" begin
        include(joinpath("composite", "test_util.jl"))
        include(joinpath("composite", "indexing.jl"))
        include(joinpath("composite", "cross.jl"))
        include(joinpath("composite", "conditioning.jl"))
        include(joinpath("composite", "product.jl"))
        include(joinpath("composite", "addition.jl"))
        include(joinpath("composite", "compose.jl"))
        include(joinpath("composite", "approximate_conditioning.jl"))
    end
end
