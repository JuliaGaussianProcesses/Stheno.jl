using Stheno: DeltaSumMean, DeltaSumKernel, LhsDeltaSumCrossKernel, RhsDeltaSumCrossKernel

@testset "delta sum" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 10, 6, 10
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

        ϕx, μx = LhsFiniteCrossKernel(EQ(), x0), FiniteMean(ConstantMean(5.1), x0)
        ϕX, μX = LhsFiniteCrossKernel(EQ(), X0), FiniteMean(ConstantMean(6.3), X0)
        g  = ConstantMean(5.0)
        μ_dsx, μ_dsX = DeltaSumMean(ϕx, μx, g), DeltaSumMean(ϕX, μX, g)
        mean_function_tests(μ_dsx, x0)
        mean_function_tests(μ_dsX, X0)

        # ϕ, k, g = EQ(), EQ(), ConstantMean(5.0)
        # k_dsx, k_dsX = DeltaSumKernel(ϕ, k, x0), DeltaSumKernel(ϕ, k, X0)

        kx, kX = FiniteKernel(EQ(), x0), FiniteKernel(EQ(), X0)
        k_dsx, k_dsX = DeltaSumKernel(ϕx, kx), DeltaSumKernel(ϕX, kX)
        kernel_tests(k_dsx, x0, x1, x2, 1e-6)
        kernel_tests(k_dsX, X0, X1, X2, 1e-6)

        kx, kX = LhsFiniteCrossKernel(EQ(), x0), LhsFiniteCrossKernel(EQ(), X0)
        k_dsx, k_dsX = LhsDeltaSumCrossKernel(ϕx, kx), LhsDeltaSumCrossKernel(ϕX, kX)
        cross_kernel_tests(k_dsx, x0, x1, x2)
        cross_kernel_tests(k_dsX, X0, X1, X2)

        kx, kX = LhsFiniteCrossKernel(EQ(), x0), LhsFiniteCrossKernel(EQ(), X0)
        k_dsx, k_dsX = RhsDeltaSumCrossKernel(kx, ϕx), RhsDeltaSumCrossKernel(kX, ϕX)
        cross_kernel_tests(k_dsx, x0, x1, x2)
        cross_kernel_tests(k_dsX, X0, X1, X2)
    end
end
