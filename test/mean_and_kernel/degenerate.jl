using Stheno: DeltaSumMean, DeltaSumKernel, LhsDeltaSumCrossKernel, RhsDeltaSumCrossKernel
using Stheno: EQ, Exp, Linear, Noise, PerEQ, OneMean

@testset "delta sum" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 10, 6, 10
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        x, X = x0, X0

        ϕ, μ, g  = EQ(), OneMean(), OneMean()
        μ_dsx, μ_dsX = DeltaSumMean(ϕ, μ, g, x), DeltaSumMean(ϕ, μ, g, X)
        differentiable_mean_function_tests(rng, μ_dsx, x0)
        differentiable_mean_function_tests(rng, μ_dsX, X0)

        k = EQ()
        k_dsx, k_dsX = DeltaSumKernel(ϕ, k, x), DeltaSumKernel(ϕ, k, X)
        differentiable_kernel_tests(rng, k_dsx, x0, x1, x2)
        differentiable_kernel_tests(rng, k_dsX, X0, X1, X2)

        k_dsx, k_dsX = LhsDeltaSumCrossKernel(ϕ, k, x), LhsDeltaSumCrossKernel(ϕ, k, X)
        differentiable_cross_kernel_tests(rng, k_dsx, x0, x1, x2)
        differentiable_cross_kernel_tests(rng, k_dsX, X0, X1, X2)

        k_dsx, k_dsX = RhsDeltaSumCrossKernel(k, ϕ, x), RhsDeltaSumCrossKernel(k, ϕ, X)
        differentiable_cross_kernel_tests(rng, k_dsx, x0, x1, x2)
        differentiable_cross_kernel_tests(rng, k_dsX, X0, X1, X2)
    end
end
