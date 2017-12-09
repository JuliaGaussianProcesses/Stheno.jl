__precompile__()

module Stheno

    using PDMats

    const ColOrRowVec = Union{AbstractVector, RowVector}

    # All covariance matrix functionality.
    include("kernel/base.jl")
    include("kernel/compose.jl")
    include("kernel/conditional.jl")
    include("kernel/transform.jl")
    include("kernel/input_transform.jl")
    include("kernel/finite.jl")

    # Covariance matrices and kernels on finite dimensional objects.
    include("covariance_matrices.jl")

    # GP stuff, including tracking and linear operator creation.
    include("gp.jl")

    # Affine transformations.
    include("lin_ops.jl")
    include("linops/addition.jl")
    include("linops/product.jl")

    # Sampling and log probability computations.
    include("sample.jl")
    include("lpdf.jl")

end # module
