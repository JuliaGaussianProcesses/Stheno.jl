using AbstractGPs, BlockArrays, LinearAlgebra, KernelFunctions, Random, Stheno, Test

using Stheno: mean, cov, var, GPC, FiniteGP, AbstractGP, BlockData, blocks, cross, ColVecs

using AbstractGPs.TestUtils: test_internal_abstractgps_interface
using AbstractGPs.Distributions: MvNormal

include("test_util.jl")
