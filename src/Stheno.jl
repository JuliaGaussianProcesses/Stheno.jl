module Stheno

    using Distributions, Distances, BlockArrays, FillArrays, Statistics, Random, Zygote,
        LinearAlgebra
    import Base: length
    import Base.Broadcast: broadcasted
    using LinearAlgebra: AbstractTriangular
    using Zygote: @adjoint, @nograd

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    const bcd = broadcasted
    const pw = pairwise

    # Various bits of utility that aren't inherently GP-related.
    include("util/zygote_rules.jl")
    include("util/covariance_matrices.jl")
    include("util/block_arrays.jl")
    include("util/abstract_data_set.jl")
    include("util/toeplitz.jl")
    include("util/eachindex_util.jl")


    # All mean function and kernel related functionality.
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/block.jl")
    include("mean_and_kernel/input_transform.jl")
    include("mean_and_kernel/degenerate.jl")
    include("mean_and_kernel/derivative.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/algebra.jl")

    # Basic Gaussian process definitions.
    include("gp/abstract_gp.jl")
    include("gp/gp.jl")
    # include("gp/block_gp.jl")

    # Affine transformations of GPs.
    include("linops/indexing.jl")
    # include("linops/addition.jl")
    # include("linops/product.jl")
    # include("linops/compose.jl")
    # include("linops/project.jl")
    # include("linops/conditioning.jl")
    # include("linops/gradient.jl")
    # # include("linops/integrate.jl")

    # # # Various stuff for convenience.
    # # include("util/model.jl")

end # module
