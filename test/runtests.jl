using Stheno, Test, BenchmarkTools, QuadGK

const check_mem = false

@testset "Stheno" begin

    @testset "mean_and_kernel" begin
        include("mean_and_kernel/mean_base.jl")
        include("mean_and_kernel/kernel_base.jl")
        include("mean_and_kernel/compose.jl")
        include("mean_and_kernel/transform.jl")
        include("mean_and_kernel/input_transform.jl")
        include("mean_and_kernel/finite.jl")
    end

    include("covariance_matrices.jl")
    include("mean_and_kernel/conditional.jl")

    include("gp.jl")

    include("lin_ops.jl")
    @testset "linops" begin
        include("linops/addition.jl")
        include("linops/product.jl")
        include("linops/integrate.jl")
    end
    
    include("sample.jl")
    include("lpdf.jl")
end
