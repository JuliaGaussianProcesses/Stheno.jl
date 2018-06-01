__precompile__(true)

module Stheno

    using Distances, BlockArrays, FillArrays
    import Base: mean, cov, show, size, length, rand, vcat, convert, promote

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}
    const AMRV{T} = Union{AbstractMatrix{T}, RowVector{T}}

    # Useful functionality for defining positive definite matrices.
    include("covariance_matrices.jl")

    # Some extensions to BlockArrays.jl.
    include("block_arrays.jl")

    # All mean function and kernel related functionality.
    include("mean_and_kernel/generic.jl")
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/cat.jl")
    # include("mean_and_kernel/transform.jl")
    include("mean_and_kernel/input_transform.jl")

    # Gaussian Process defintions.
    include("gp.jl")

    # Affine transformations of GPs.
    include("linops/indexing.jl")
    include("linops/conditioning.jl")
    include("linops/addition.jl")
    include("linops/product.jl")
    # include("linops/integrate.jl")

    # Code to make Stheno work with Turing.
    include("turing_util.jl")
end # module
