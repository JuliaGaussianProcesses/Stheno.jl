module Stheno

    using Distributions, Distances, BlockArrays, FillArrays, Statistics, Random, Zygote,
        LinearAlgebra
    import Base: length, map
    import Base.Broadcast: broadcasted, materialize, broadcast_shape
    import Statistics: mean, cov
    using LinearAlgebra: AbstractTriangular
    using Zygote: @adjoint, @nograd, @showgrad
    using BlockArrays: _BlockArray
    import LinearAlgebra: cholesky

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    function elementwise end

    const pw = pairwise
    const ew = elementwise

    Zygote.@nograd broadcast_shape

    # Various bits of utility that aren't inherently GP-related. A lot of this is very
    # type-piratic.
    include("util/zygote_rules.jl")
    include("util/covariance_matrices.jl")
    include("util/block_arrays/dense.jl")
    include("util/block_arrays/diagonal.jl")
    include("util/block_arrays/triangular.jl")
    include("util/abstract_data_set.jl")
    include("util/toeplitz.jl")
    include("util/fillarrays.jl")
    include("util/proper_type_piracy.jl")

    # All mean function and kernel related functionality.
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/block.jl")
    include("mean_and_kernel/input_transform.jl")
    # include("mean_and_kernel/derivative.jl")
    include("mean_and_kernel/conditioning/exact.jl")
    include("mean_and_kernel/conditioning/titsias.jl")
    include("mean_and_kernel/algebra.jl")
    include("mean_and_kernel/util.jl")

    # Basic Gaussian process definitions.
    include("gp/gp.jl")
    include("gp/block_gp.jl")
    include("gp/finite_gp.jl")

    # Affine transformations of GPs.
    include("linops/indexing.jl")
    include("linops/conditioning.jl")
    include("linops/approximate_conditioning.jl")
    include("linops/product.jl")
    include("linops/addition.jl")
    include("linops/compose.jl")
    # include("linops/gradient.jl")
    # # include("linops/integrate.jl")

    # Various stuff for convenience.
    include("util/model.jl")
end # module
