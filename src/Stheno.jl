module Stheno

    using Reexport

    using AbstractGPs
    using BlockArrays
    using ChainRulesCore
    using Distributions
    using FillArrays
    @reexport using KernelFunctions
    using LinearAlgebra
    using Random
    using Requires
    using Zygote
    using ZygoteRules

    import Base.Broadcast: broadcasted

    using AbstractGPs: AbstractGP, FiniteGP, GP
    import AbstractGPs:
        mean,
        cov,
        cov_diag,
        mean_and_cov,
        mean_and_cov_diag,
        rand,
        logpdf,
        elbo,
        dtc

    using MacroTools: @capture, combinedef, postwalk, splitdef

    using ZygoteRules: @adjoint

    const AV{T} = AbstractVector{T}

    # Various bits of utility that aren't inherently GP-related. Often very type-piratic.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "covariance_matrices.jl"))
    include(joinpath("util", "block_arrays", "dense.jl"))
    include(joinpath("util", "block_arrays", "diagonal.jl"))
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

    function __init__()
        @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
            include(joinpath("flux", "neural_kernel_network.jl"))
        end
    end

    include(joinpath("deprecate.jl"))

    export wrap, BlockData, GPC, GPPPInput, @gppp
    export elbo, dtc
    export ∘, select, stretch, periodic, shift
end # module
