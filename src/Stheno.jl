__precompile__(true)

module Stheno

    using LinearAlgebra, Random, Distances, BlockArrays
    import Base: mean, cov, show, size, length, rand, vcat, convert, promote

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    # Some extensions to BlockArrays.jl.
    include("block_arrays.jl")

    # Useful functionality for defining positive definite matrices.
    include("covariance_matrices.jl")

    # All mean function and kernel related functionality.
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/cat.jl")
    # include("mean_and_kernel/transform.jl")
    # include("mean_and_kernel/input_transform.jl")

    # Gaussian Process defintions.
    include("gp.jl")

    # Affine transformations of GPs.
    include("linops/indexing.jl")
    include("linops/conditioning.jl")
    include("linops/addition.jl")
    include("linops/product.jl")
    # include("linops/integrate.jl")
end # module
