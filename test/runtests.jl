using AbstractGPs
using BlockArrays
using Documenter
using FiniteDifferences
using Flux
using LinearAlgebra
using KernelFunctions
using Random
using Statistics
using Stheno
using Test
using TimerOutputs
using Zygote

using Stheno: mean, cov, cov_diag, GPC, AV

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
        @testset "block_arrays" begin
            include(joinpath("util", "block_arrays", "test_util.jl"))
            include(joinpath("util", "block_arrays", "dense.jl"))
            include(joinpath("util", "block_arrays", "diagonal.jl"))
        end
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

    println("flux:")
    @timedtestset "flux" begin
        include(joinpath("flux", "neural_kernel_network.jl"))
    end

    println("doctests")
    @timedtestset "doctests" begin
        DocMeta.setdocmeta!(
            Stheno,
            :DocTestSetup,
            :(using AbstractGPs, Stheno, Random, Documenter, LinearAlgebra);
            recursive=true,
        )
        doctest(Stheno)
    end
end

display(to)
