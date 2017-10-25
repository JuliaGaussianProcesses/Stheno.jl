module Stheno

    using PDMats

    const ColOrRowVec = Union{AbstractVector, RowVector}

    # All covariance matrix functionality.
    include("kernel/base.jl")
    include("kernel/compose.jl")
    include("kernel/transform.jl")
    include("kernel/input_transform.jl")

    # GP stuff, including tracking and linear operator creation.
    include("gp.jl")
    include("lin_ops.jl")

    # Covariance matrices and kernels on finite dimensional objects.
    include("covariance_matrices.jl")

end # module
