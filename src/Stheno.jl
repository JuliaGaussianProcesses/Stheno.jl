module Stheno

    using Reexport

    using AbstractGPs
    using BlockArrays
    using ChainRulesCore
    using FillArrays
    @reexport using KernelFunctions
    using LinearAlgebra
    using Random
    using Zygote
    using ZygoteRules

    import Base.Broadcast: broadcasted

    using AbstractGPs: AbstractGP, FiniteGP, GP
    import AbstractGPs:
        mean,
        cov,
        var,
        mean_and_cov,
        mean_and_var,
        rand,
        logpdf,
        elbo,
        dtc,
        posterior,
        marginals

    using MacroTools: @capture, combinedef, postwalk, splitdef

    const AV{T} = AbstractVector{T}

    # Various bits of utility that aren't inherently GP-related. Often very type-piratic.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "covariance_matrices.jl"))
    include(joinpath("util", "block_arrays.jl"))

    # New AbstractGPs and associated utility.
    include(joinpath("gp", "util.jl"))
    include(joinpath("gp", "atomic_gp.jl"))
    include(joinpath("gp", "derived_gp.jl"))
    include(joinpath("gp", "sparse_finite_gp.jl"))

    # Composite GPs, constructed via affine transformation of CompositeGPs and GPs.
    include(joinpath("affine_transformations", "cross.jl"))
    include(joinpath("affine_transformations", "product.jl"))
    include(joinpath("affine_transformations", "addition.jl"))
    include(joinpath("affine_transformations", "compose.jl"))

    include("gaussian_process_probabilistic_programme.jl")

    export atomic, BlockData, GPC, GPPPInput, @gppp
    export ∘, select, stretch, periodic, shift
    export cov_diag, mean_and_cov_diag
end # module
