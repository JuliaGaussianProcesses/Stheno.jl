using Revise
using Stheno, Base.Test, QuadGK, BlockArrays, FillArrays

@testset "Stheno" begin

    include("util/covariance_matrices.jl")
    include("util/woodbury.jl")
    include("util/block_arrays.jl")

    include("test_util.jl")
    include("mean_and_kernel/generic.jl")
    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/cat.jl")
    # include("mean_and_kernel/transform.jl")
    include("mean_and_kernel/input_transform.jl")
    include("mean_and_kernel/degenerate.jl")

    include("gp/abstract_gp.jl")
    include("gp/gp.jl")
    include("gp/joint_gp.jl")

    include("linops/indexing.jl")
    include("linops/addition.jl")
    include("linops/product.jl")
    include("linops/integrate.jl")
    include("linops/project.jl")
    include("linops/conditioning.jl")
end
