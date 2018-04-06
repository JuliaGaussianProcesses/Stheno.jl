module Stheno

    using PDMats, LinearAlgebra, Random, Distances
    import Base: mean, cov, show, size, +, *, isfinite, rand

    const ColOrRowVec = Union{AbstractVector, RowVector}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    # Useful functionality for defining positive definite matrices.
    include("covariance_matrices.jl")

    # All mean function and kernel related functionality.
    """
        MeanFunction
    """
    abstract type MeanFunction end

    """
        CrossKernel

    Supertype for all cross-Kernels. There are binary functions, but are not valid Mercer
    kernels as they are not in general symmetric positive semi-definite.
    """
    abstract type CrossKernel end

    """
        Kernel <: CrossKernel

    Supertype for all (valid Mercer) Kernels.
    """
    abstract type Kernel <: CrossKernel end

    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/compose.jl")
    # include("mean_and_kernel/transform.jl")
    # include("mean_and_kernel/input_transform.jl")
    

    # GP stuff, including tracking and linear operator creation.
    include("gp.jl")

    # # Affine transformations.
    # include("lin_ops.jl")
    # include("linops/addition.jl")
    # include("linops/product.jl")
    # # include("linops/integrate.jl")
end # module
