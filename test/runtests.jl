using Stheno
using Base.Test
using BenchmarkTools

const check_mem = false

@testset "Stheno" begin

    # @testset "kernel" begin
    #     include("kernel/base.jl")
    #     include("kernel/compose.jl")
    #     include("kernel/transform.jl")
    #     include("kernel/input_transform.jl")
    #     include("kernel/finite.jl")
    # end

    # include("covariance_matrices.jl")
    # include("kernel/conditional.jl")
    # include("gp.jl")

    # @testset "linops" begin
    #     include("linops/addition.jl")
    #     include("linops/product.jl")
    # end
    include("lin_ops.jl")

    include("sample.jl")
    include("lpdf.jl")
end
