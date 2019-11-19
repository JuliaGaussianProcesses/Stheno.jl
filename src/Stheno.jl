module Stheno

    using Distributions, Distances, BlockArrays, Statistics, Random,
        LinearAlgebra, Zygote
    import Base: length, map
    import Base.Broadcast: broadcasted, materialize, broadcast_shape
    import Statistics: mean, cov
    using LinearAlgebra: AbstractTriangular
    using ZygoteRules: @adjoint
    using Zygote: @nograd
    using BlockArrays: _BlockArray
    import LinearAlgebra: cholesky, cross
    import Distances: pairwise, colwise

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    const BlockLowerTriangular{T} = LowerTriangular{T, <:BlockMatrix{T}}
    const BlockUpperTriangular{T} = UpperTriangular{T, <:BlockMatrix{T}}
    const BlockTriangular{T} = Union{BlockLowerTriangular{T}, BlockUpperTriangular{T}}

    function elementwise end

    const pw = pairwise
    const ew = elementwise

    # Various bits of utility that aren't inherently GP-related. Often very type-piratic.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "covariance_matrices.jl"))
    include(joinpath("util", "block_arrays", "dense.jl"))
    include(joinpath("util", "block_arrays", "diagonal.jl"))
    include(joinpath("util", "block_arrays", "triangular.jl"))
    include(joinpath("util", "abstract_data_set.jl"))
    include(joinpath("util", "distances.jl"))
    include(joinpath("util", "proper_type_piracy.jl"))

    # Supertype for GPs.
    include("abstract_gp.jl")

    # Atomic GP objects.
    include(joinpath("gp", "mean.jl"))
    include(joinpath("gp", "kernel.jl"))
    include(joinpath("gp", "gp.jl"))

    # Composite GPs, constructed via affine transformation of CompositeGPs and GPs.
    include(joinpath("composite", "composite_gp.jl"))
    include(joinpath("composite", "indexing.jl"))
    include(joinpath("composite", "cross.jl"))
    include(joinpath("composite", "conditioning.jl"))
    include(joinpath("composite", "approximate_conditioning.jl"))
    include(joinpath("composite", "product.jl"))
    include(joinpath("composite", "addition.jl"))
    include(joinpath("composite", "compose.jl"))
    # include(joinpath("composite", "gradient.jl"))
    # include(joinpath("composite", "integrate.jl"))

    # Various stuff for convenience.
    include(joinpath("util", "model.jl"))
    include(joinpath("util", "plotting.jl"))
end # module
