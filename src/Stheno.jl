__precompile__(true)

module Stheno

    using PDMats, LinearAlgebra, Random, Distances, BlockArrays
    import Base: mean, cov, show, size, length, +, *, isfinite, rand, vcat, convert, promote

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    # Useful functionality for defining positive definite matrices.
    include("covariance_matrices.jl")

    # Some extensions to BlockArrays.jl.
    include("block_arrays.jl")

    # All mean function and kernel related functionality.
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/cat.jl")
    # include("mean_and_kernel/transform.jl")
    # include("mean_and_kernel/input_transform.jl")

    # GP stuff, including tracking and linear operator creation.
    include("gp.jl")

    # Affine transformations.
    include("lin_ops.jl")
    # include("linops/addition.jl")
    # include("linops/product.jl")
    # include("linops/integrate.jl")
end # module
