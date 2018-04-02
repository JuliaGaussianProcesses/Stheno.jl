module Stheno

    using PDMats, LinearAlgebra, Random

    const ColOrRowVec = Union{AbstractVector, RowVector}

    # All covariance matrix functionality.
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/transform.jl")
    include("mean_and_kernel/input_transform.jl")
    include("mean_and_kernel/finite.jl")

    # Covariance matrices and kernels on finite dimensional objects.
    include("covariance_matrices.jl")

    # GP stuff, including tracking and linear operator creation.
    include("gp.jl")

    # Affine transformations.
    include("lin_ops.jl")
    include("linops/addition.jl")
    include("linops/product.jl")
    include("linops/integrate.jl")

    # Sampling and log probability computations.
    include("sample.jl")
    include("lpdf.jl")

end # module
