module Stheno

    using Distributions, Distances, BlockArrays, FillArrays, Statistics, Random, Zygote,
        LinearAlgebra
    import Base: length, map
    import Base.Broadcast: broadcasted, materialize, broadcast_shape
    import Statistics: mean, cov
    using LinearAlgebra: AbstractTriangular
    using Zygote: @adjoint, @nograd, @showgrad, hook
    using BlockArrays: _BlockArray
    import LinearAlgebra: cholesky, cross

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    const BlockLowerTriangular{T} = LowerTriangular{T, <:BlockMatrix{T}}
    const BlockUpperTriangular{T} = UpperTriangular{T, <:BlockMatrix{T}}
    const BlockTriangular{T} = Union{BlockLowerTriangular{T}, BlockUpperTriangular{T}}


    function elementwise end

    const pw = pairwise
    const ew = elementwise

    Zygote.@nograd broadcast_shape


    showtype(x) = (@show typeof(x); x)
    showsize(x) = (@show size(x); x)


    # Various bits of utility that aren't inherently GP-related. A lot of this is very
    # type-piratic.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "covariance_matrices.jl"))
    include(joinpath("util", "block_arrays", "dense.jl"))
    include(joinpath("util", "block_arrays", "diagonal.jl"))
    include(joinpath("util", "block_arrays", "triangular.jl"))
    include(joinpath("util", "abstract_data_set.jl"))
    include(joinpath("util", "toeplitz.jl"))
    include(joinpath("util", "fillarrays.jl"))
    include(joinpath("util", "proper_type_piracy.jl"))

    # All mean function and kernel related functionality.
    include(joinpath("mean_and_kernel", "mean.jl"))
    include(joinpath("mean_and_kernel", "kernel.jl"))
    include(joinpath("mean_and_kernel", "compose.jl"))
    include(joinpath("mean_and_kernel", "block.jl"))
    include(joinpath("mean_and_kernel", "input_transform.jl"))
    # include(joinpath("mean_and_kernel", "derivative.jl"))
    include(joinpath("mean_and_kernel", "conditioning", "exact.jl"))
    include(joinpath("mean_and_kernel", "conditioning", "titsias.jl"))
    include(joinpath("mean_and_kernel", "algebra.jl"))
    include(joinpath("mean_and_kernel", "util.jl"))

    # Basic Gaussian process definitions.
    include(joinpath("gp", "gp.jl"))
    include(joinpath("gp", "finite_gp.jl"))

    # Affine transformations of GPs.
    include(joinpath("linops", "indexing.jl"))
    include(joinpath("linops", "cross.jl"))
    include(joinpath("linops", "conditioning.jl"))
    include(joinpath("linops", "approximate_conditioning.jl"))
    include(joinpath("linops", "product.jl"))
    include(joinpath("linops", "addition.jl"))
    include(joinpath("linops", "compose.jl"))
    # include(joinpath("linops", "gradient.jl"))
    # # include(joinpath("linops", "integrate.jl"))

    # Various stuff for convenience.
    include(joinpath("util", "model.jl"))
end # module
