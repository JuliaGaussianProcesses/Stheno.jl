using Stheno: DeltaSumMean, DeltaSumKernel, LhsDeltaSumCrossKernel, RhsDeltaSumCrossKernel

@testset "delta sum" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 10, 6, 10
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = MatData(randn(rng, D, N)), MatData(randn(rng, D, N)), MatData(randn(rng, D, N′))

        ϕx, μx = LhsFiniteCrossKernel(EQ(), x0), FiniteMean(ConstantMean(5.1), x0)
        ϕX, μX = LhsFiniteCrossKernel(EQ(), X0), FiniteMean(ConstantMean(6.3), X0)
        g  = ConstantMean(5.0)
        μ_dsx, μdsX = DeltaSumMean(ϕx, μx, g), DeltaSumMean(ϕX, μX, g)
        mean_function_tests(μ_dsx, x0)
        mean_function_tests(μ_dsx, X0)



        ϕ, k, g = EQ(), EQ(), ConstantMean(5.0)
        k_dsx, k_dsX = DeltaSumKernel(ϕ, k, x0), DeltaSumKernel(ϕ, k, X0)

        kernel_tests(k_dsx, x0, x1, x2, 1e-6)
        kernel_tests(k_dsX, X0, X1, X2, 1e-6)

        k_dsx, k_dsX = LhsDeltaSumCrossKernel(ϕ, k, x0), LhsDeltaSumCrossKernel(ϕ, k, X0)
        cross_kernel_tests(k_dsx, x0, x1, x2)
        cross_kernel_tests(k_dsX, X0, X1, X2)

        k_dsx, k_dsX = RhsDeltaSumCrossKernel(k, x0, ϕ), RhsDeltaSumCrossKernel(k, X0, ϕ)
        cross_kernel_tests(k_dsx, x0, x1, x2)
        cross_kernel_tests(k_dsX, X0, X1, X2)
    end
end
