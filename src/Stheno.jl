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
        approx_posterior,
        marginals

    using MacroTools: @capture, combinedef, postwalk, splitdef

    const AV{T} = AbstractVector{T}

    # Various bits of utility that aren't inherently GP-related. Often very type-piratic.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "covariance_matrices.jl"))
    include(joinpath("util", "dense.jl"))
    include(joinpath("util", "abstract_data_set.jl"))
    include(joinpath("util", "proper_type_piracy.jl"))

    # Supertype for GPs.
    include("abstract_gp.jl")

    # Atomic GP objects.
    include(joinpath("gp", "gp.jl"))

    # Composite GPs, constructed via affine transformation of CompositeGPs and GPs.
    include(joinpath("composite", "composite_gp.jl"))
    include(joinpath("composite", "cross.jl"))
    include(joinpath("composite", "product.jl"))
    include(joinpath("composite", "addition.jl"))
    include(joinpath("composite", "compose.jl"))
    # include(joinpath("composite", "gradient.jl"))
    # include(joinpath("composite", "integrate.jl"))

    # Gaussian Process Probabilistic Programme object which implements the AbstractGPs API.
    include("gaussian_process_probabilistic_programme.jl")

    # Sparse GP hack to make pseudo-point approximations play nicely with Turing.jl.
    include("sparse_finite_gp.jl")

    include("deprecate.jl")

    export wrap, BlockData, GPC, GPPPInput, @gppp
    export ∘, select, stretch, periodic, shift
    export cov_diag, mean_and_cov_diag
end # module
