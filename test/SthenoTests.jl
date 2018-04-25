module SthenoTests
    using Stheno, Test, BenchmarkTools, QuadGK, Random, LinearAlgebra, BlockArrays
    const check_mem = false

    include("test_util.jl")

    include("covariance_matrices.jl")
    include("block_arrays.jl")

    include("mean_and_kernel/mean.jl")
    include("mean_and_kernel/kernel.jl")
    include("mean_and_kernel/compose.jl")
    include("mean_and_kernel/conditional.jl")
    include("mean_and_kernel/finite.jl")
    include("mean_and_kernel/cat.jl")
    include("mean_and_kernel/transform.jl")
    include("mean_and_kernel/input_transform.jl")

    include("gp.jl")

    include("linops/indexing.jl")
    include("linops/conditioning.jl")
    include("linops/addition.jl")
    include("linops/product.jl")
    include("linops/integrate.jl")

    function run()
        @testset "Stheno" begin
            covariance_matrices_tests()
            block_arrays_tests()

            @testset "mean_and_kernel" begin
                mean_and_kernel_mean_tests()
                mean_and_kernel_kernel_tests()
                mean_and_kernel_compose_tests()
                mean_and_kernel_conditional_tests()
                mean_and_kernel_finite_tests()
                mean_and_kernel_cat_tests()
                # mean_and_kernel_transform_tests()
                # mean_and_kernel_input_transform_tests()
            end

            gp_tests()

            @testset "LinOps" begin
                linops_indexing_tests()
                linops_conditioning_tests()
                linops_addition_tests()
                linops_product_tests()
                # linops_integrate_tests()
            end
        end
    end
end
