module Stheno

    using
        BlockArrays,
        ChainRulesCore,
        Distributions,
        Distances,
        FillArrays,
        LinearAlgebra,
        Statistics,
        Random,
        Requires,
        Zygote,
        ZygoteRules

    import Base: length, map
    import Base.Broadcast: broadcasted, materialize, broadcast_shape
    import ChainRulesCore: rrule
    import Statistics: mean, cov
    using ZygoteRules: @adjoint
    using Zygote: @nograd
    import LinearAlgebra: cholesky, cross
    import Distances: pairwise, colwise

    const AV{T} = AbstractVector{T}
    const AM{T} = AbstractMatrix{T}
    const AVM{T} = AbstractVecOrMat{T}

    function elementwise end

    const pw = pairwise
    const ew = elementwise

    # Various bits of utility that aren't inherently GP-related. Often very type-piratic.
    include(joinpath("util", "zygote_rules.jl"))
    include(joinpath("util", "covariance_matrices.jl"))
    include(joinpath("util", "block_arrays", "dense.jl"))
    include(joinpath("util", "block_arrays", "diagonal.jl"))
    include(joinpath("util", "abstract_data_set.jl"))
    include(joinpath("util", "distances.jl"))
    include(joinpath("util", "proper_type_piracy.jl"))

    # Supertype for GPs.
    include("abstract_gp.jl")

    # Atomic GP objects.
    include(joinpath("gp", "mean.jl"))
    include(joinpath("gp", "kernel.jl"))
    include(joinpath("gp", "gp.jl"))

    # Composite GPs, constructed via affine transformation of CompositeGPs and GPs.
    include(joinpath("composite", "composite_gp.jl"))
    include(joinpath("composite", "indexing.jl"))
    include(joinpath("composite", "cross.jl"))
    include(joinpath("composite", "conditioning.jl"))
    include(joinpath("composite", "approximate_conditioning.jl"))
    include(joinpath("composite", "product.jl"))
    include(joinpath("composite", "addition.jl"))
    include(joinpath("composite", "compose.jl"))
    # include(joinpath("composite", "gradient.jl"))
    # include(joinpath("composite", "integrate.jl"))

    # Various stuff for convenience.
    include(joinpath("util", "model.jl"))
    include(joinpath("util", "plotting.jl"))

    function __init__()
        @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
            include(joinpath("flux", "add_functor.jl"))
            include(joinpath("flux", "neural_kernel_network.jl"))
        end
    end

end # module
