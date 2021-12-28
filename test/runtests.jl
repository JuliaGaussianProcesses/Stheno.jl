using Stheno

# Dependencies that are also used in Stheno that are not in the standard library.
# This is a hack to prevent CompatHelper from trying to maintain a version number for these
# packages in both Stheno and test.
using Stheno.AbstractGPs
using Stheno.BlockArrays
using Stheno.KernelFunctions
using Stheno.Zygote

# Dependencies that are test-specific.
using Documenter
using FiniteDifferences
using LinearAlgebra
using Random
using Statistics
using Test
using TimerOutputs

using Stheno:
    mean,
    cov,
    var,
    GPC,
    AV,
    FiniteGP,
    AbstractGP,
    BlockData,
    blocks,
    cross,
    ColVecs,
    Xt_invA_Y,
    Xt_invA_X,
    diag_At_A,
    diag_At_B,
    diag_Xt_invA_X,
    diag_Xt_invA_Y

using Stheno.AbstractGPs.TestUtils: test_internal_abstractgps_interface
using Stheno.AbstractGPs.Distributions: MvNormal
using FiniteDifferences: j′vp

const to = TimerOutput()

macro timedtestset(name, code)
    return esc(:(@timeit to $name @testset $name $code))
end

include("test_util.jl")

@testset "Stheno" begin

    println("util:")
    @timedtestset "util" begin
        include(joinpath("util", "zygote_rules.jl"))
        include(joinpath("util", "covariance_matrices.jl"))
        include(joinpath("util", "block_arrays.jl"))
        include(joinpath("util", "abstract_data_set.jl"))
    end

    println("gp:")
    @timedtestset "gp" begin
        include(joinpath("gp", "gp.jl"))
    end

    println("composite:")
    @timedtestset "composite" begin
        include(joinpath("composite", "test_util.jl"))
        include(joinpath("composite", "cross.jl"))
        include(joinpath("composite", "product.jl"))
        include(joinpath("composite", "addition.jl"))
        include(joinpath("composite", "compose.jl"))
    end

    println("abstract_gp:")
    @timedtestset "abstract_gp" begin
        include("abstract_gp.jl")
    end

    println("gaussian_process_probabilistic_programme:")
    include("gaussian_process_probabilistic_programme.jl")

    println("sparse_finite_gp:")
    include("sparse_finite_gp.jl")

    println("doctests")
    @timedtestset "doctests" begin
        DocMeta.setdocmeta!(
            Stheno,
            :DocTestSetup,
            :(using Stheno.AbstractGPs, Stheno, Random, Documenter, LinearAlgebra);
            recursive=true,
        )
        doctest(Stheno)
    end
end

display(to)
