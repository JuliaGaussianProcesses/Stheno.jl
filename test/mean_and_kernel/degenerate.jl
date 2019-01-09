using Stheno: DeltaSumMean, DeltaSumKernel, LhsDeltaSumCrossKernel, RhsDeltaSumCrossKernel

@testset "delta sum" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 10, 6, 10
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))

        ϕx, μx = LhsFiniteCrossKernel(EQ(), x0), FiniteMean(OneMean(), x0)
        ϕX, μX = LhsFiniteCrossKernel(EQ(), X0), FiniteMean(OneMean(), X0)
        g  = OneMean()
        μ_dsx, μ_dsX = DeltaSumMean(ϕx, μx, g), DeltaSumMean(ϕX, μX, g)
        differentiable_mean_function_tests(rng, μ_dsx, x0)
        differentiable_mean_function_tests(rng, μ_dsX, X0)

        kx, kX = FiniteKernel(EQ(), x0), FiniteKernel(EQ(), X0)
        k_dsx, k_dsX = DeltaSumKernel(ϕx, kx), DeltaSumKernel(ϕX, kX)
        differentiable_kernel_tests(rng, k_dsx, x0, x1, x2)
        differentiable_kernel_tests(rng, k_dsX, X0, X1, X2)

        kx, kX = LhsFiniteCrossKernel(EQ(), x0), LhsFiniteCrossKernel(EQ(), X0)
        k_dsx, k_dsX = LhsDeltaSumCrossKernel(ϕx, kx), LhsDeltaSumCrossKernel(ϕX, kX)
        differentiable_cross_kernel_tests(rng, k_dsx, x0, x1, x2)
        differentiable_cross_kernel_tests(rng, k_dsX, X0, X1, X2)

        kx, kX = LhsFiniteCrossKernel(EQ(), x0), LhsFiniteCrossKernel(EQ(), X0)
        k_dsx, k_dsX = RhsDeltaSumCrossKernel(kx, ϕx), RhsDeltaSumCrossKernel(kX, ϕX)
        differentiable_cross_kernel_tests(rng, k_dsx, x0, x1, x2)
        differentiable_cross_kernel_tests(rng, k_dsX, X0, X1, X2)
    end
end
