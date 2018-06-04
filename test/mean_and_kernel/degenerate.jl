using Stheno: DegenerateKernel, DegenerateCrossKernel, binary_obswise, pairwise

@testset "degenerate" begin

    # DegenerateKernel tests.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        x0, x1, x2 = randn(rng, N), randn(rng, N), randn(rng, N′)
        X0, X1, X2 = randn(rng, D, N), randn(rng, D, N), randn(rng, D, N′)

        A_ = randn(N, N)
        A = LazyPDMat(A_' * A_, 1e-12)
        kfgx0, kfgX0 = LhsFiniteCrossKernel(EQ(), x0), LhsFiniteCrossKernel(EQ(), X0)
        kx0, kX0 = DegenerateKernel(A, kfgx0), DegenerateKernel(A, kfgX0)

        kernel_tests(kx0, x0, x1, x2)
        kernel_tests(kX0, X0, X1, X2)
        @test pairwise(kx0, x0, x1) ≈ pairwise(kfgx0, :, x0)' * A * pairwise(kfgx0, :, x1)
        @test pairwise(kX0, X0, X1) ≈ pairwise(kfgX0, :, X0)' * A * pairwise(kfgX0, :, X1)

        kfhx1, kfhX1 = LhsFiniteCrossKernel(EQ(), x1), LhsFiniteCrossKernel(EQ(), X1)
        kx01 = DegenerateCrossKernel(kfgx0, A, kfhx1)
        kX01 = DegenerateCrossKernel(kfgX0, A, kfhX1)

        cross_kernel_tests(kx01, x0, x1, x2)
        cross_kernel_tests(kX01, X0, X1, X2)
        @test pairwise(kx01, x0, x1) ≈ pairwise(kfgx0, :, x0)' * A * pairwise(kfhx1, :, x1)
        @test pairwise(kX01, X0, X1) ≈ pairwise(kfgX0, :, X0)' * A * pairwise(kfhX1, :, X1)
    end
end
