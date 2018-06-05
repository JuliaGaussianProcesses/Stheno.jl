using Stheno: DeltaSumMean, DeltaSumKernel, LhsDeltaSumCrossKernel, RhsDeltaSumCrossKernel

@testset "delta sum" begin

    let
        rng, N, N′, D = MersenneTwister(123456), 10, 6, 10
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

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
