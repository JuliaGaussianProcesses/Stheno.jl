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
    end

    println("gp:")
    @timedtestset "gp" begin
        include(joinpath("gp", "util.jl"))
        include(joinpath("gp", "atomic_gp.jl"))
        include(joinpath("gp", "derived_gp.jl"))
        include(joinpath("gp", "sparse_finite_gp.jl"))
    end

    println("affine_transformations:")
    @timedtestset "affine_transformations" begin
        include(joinpath("affine_transformations", "test_util.jl"))
        include(joinpath("affine_transformations", "addition.jl"))
        include(joinpath("affine_transformations", "compose.jl"))
        include(joinpath("affine_transformations", "product.jl"))
    end

    include(joinpath("affine_transformations", "cross.jl"))
    include("gaussian_process_probabilistic_programme.jl")

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
