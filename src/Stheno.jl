module Stheno

    # Users generally need access to the functionality from both of these packages.
    using Reexport
    @reexport using AbstractGPs
    @reexport using KernelFunctions

    using BlockArrays
    using ChainRulesCore
    using LinearAlgebra
    using Random

    import Base.Broadcast: broadcasted

    using AbstractGPs: AbstractGP, FiniteGP
    import AbstractGPs: mean, cov, var

    using MacroTools: @capture, combinedef, postwalk, splitdef

    const AV{T} = AbstractVector{T}

    import ChainRulesCore: rrule

    # A couple of AbstractVector subtypes useful for expressing structure in inputs
    # regularly found in GPPPs.
    include("input_collection_types.jl")

    # AbstractGP subtypes and associated utility.
    include(joinpath("gp", "util.jl"))
    include(joinpath("gp", "atomic_gp.jl"))
    include(joinpath("gp", "derived_gp.jl"))
    include(joinpath("gp", "sparse_finite_gp.jl"))

    # Affine transformation library. Each file contains one / a couple of closely-related
    # affine transformations. Consequently, the code in each file can be understood
    # independently of the code in each other file.
    include(joinpath("affine_transformations", "cross.jl"))
    include(joinpath("affine_transformations", "addition.jl"))
    include(joinpath("affine_transformations", "compose.jl"))
    include(joinpath("affine_transformations", "product.jl"))
    include(joinpath("affine_transformations", "additive_gp.jl"))

    # AbstractGP subtype which groups together other AbstractGP subtypes.
    include("gaussian_process_probabilistic_programme.jl")

    export atomic, BlockData, GPC, GPPPInput, @gppp
    export ∘, select, stretch, periodic, shift, additive_gp
    export SparseFiniteGP

end # module
