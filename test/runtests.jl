using Revise
using Revise: track
using Stheno, Base.Test, BenchmarkTools, QuadGK, BlockArrays

# Track test files to avoid needlessly reloading.
track.([
    "block_arrays.jl",
    "covariance_matrices.jl"
])

include("test_util.jl")

@testset "Stheno" begin

    include("block_arrays.jl")
    # include("covariance_matrices.jl")

    # include("mean_and_kernel/mean.jl")
    # include("mean_and_kernel/kernel.jl")
    # include("mean_and_kernel/compose.jl")
    # include("mean_and_kernel/conditional.jl")
    # include("mean_and_kernel/finite.jl")
    # include("mean_and_kernel/cat.jl")
    # include("mean_and_kernel/transform.jl")
    # include("mean_and_kernel/input_transform.jl")

    # include("gp.jl")

    # include("linops/indexing.jl")
    # include("linops/conditioning.jl")
    # include("linops/addition.jl")
    # include("linops/product.jl")
    # include("linops/integrate.jl")
end
