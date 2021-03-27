module Stheno

    using Reexport

    using AbstractGPs
    using BlockArrays
    using ChainRulesCore
    using Distributions
    using Distances
    using FillArrays
    @reexport using KernelFunctions
    using LinearAlgebra
    using Statistics
    using Random
    using Requires
    using Zygote
    using ZygoteRules

    import Base: length, map
    import Base.Broadcast: broadcasted, materialize, broadcast_shape
    import ChainRulesCore: rrule
    import Statistics: mean, cov

    import LinearAlgebra: cholesky, cross
    import Distances: pairwise, colwise

    using AbstractGPs: AbstractGP, GP, FiniteGP
    import AbstractGPs: cov_diag, rand, logpdf, elbo, dtc, mean_and_cov, mean_and_cov_diag

    using ZygoteRules: @adjoint
    using Zygote: @nograd

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

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
    include(joinpath("composite", "conditioning.jl"))
    include(joinpath("composite", "approximate_conditioning.jl"))
    include(joinpath("composite", "product.jl"))
    include(joinpath("composite", "addition.jl"))
    include(joinpath("composite", "compose.jl"))
    # include(joinpath("composite", "gradient.jl"))
    # include(joinpath("composite", "integrate.jl"))

    # approximate inference
    include("approximate_inference.jl")

    # Various stuff for convenience.
    include(joinpath("util", "model.jl"))

    function __init__()
        @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
            include(joinpath("flux", "neural_kernel_network.jl"))
        end
    end

    export wrap, GP

end # module
