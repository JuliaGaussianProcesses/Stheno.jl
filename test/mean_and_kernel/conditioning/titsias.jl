using Random, FillArrays
using Stheno: OneMean, EQ, PPC, TitsiasMean, TitsiasKernel, TitsiasCrossKernel, CustomMean

@testset "titsias" begin

    @testset "TitsiasMean" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x = collect(range(-3.0, 3.0; length=N))
        z = collect(range(-3.0, 3.0; length=M))

        m̂ε = randn(M)
        A = randn(rng, M, N)
        S = PPC(cholesky(A * A' + I), cholesky(pw(EQ(), z)).U)
        m = TitsiasMean(S, CustomMean(sin), EQ(), m̂ε, z)

        differentiable_mean_function_tests(rng, m, x)
    end

    @testset "TitsiasKernel" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x0 = collect(range(-3.0, stop=3.0, length=N))
        x1 = collect(range(-3.5, stop=2.5, length=N))
        x2 = collect(range(-2.5, stop=3.5, length=N′))
        z = collect(range(-3.0, 3.0; length=M))

        A = randn(rng, M, N)
        S = PPC(cholesky(A * A' + I), cholesky(pw(EQ(), z)).U)
        k = TitsiasKernel(S, EQ(), EQ(), z)

        differentiable_kernel_tests(rng, k, x0, x1, x2)
    end

    @testset "TitsiasCrossKernel" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x0 = collect(range(-3.0, stop=3.0, length=N))
        x1 = collect(range(-3.5, stop=2.5, length=N))
        x2 = collect(range(-2.5, stop=3.5, length=N′))
        z = collect(range(-3.0, 3.0; length=M))

        A = randn(rng, M, N)
        S = PPC(cholesky(A * A' + I), cholesky(pw(EQ(), z)).U)
        k = TitsiasCrossKernel(S, EQ(), EQ(), EQ(), z)

        differentiable_cross_kernel_tests(rng, k, x0, x1, x2)
    end
end
