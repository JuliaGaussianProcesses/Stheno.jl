using Random
using Stheno: OneMean, EQ, PPC, ApproxCondMean, ApproxCondKernel, ApproxCondCrossKernel,
    CustomMean

@testset "titsias" begin

    function construct_cache(N, rng, z)
        M = length(z)
        m̂ε = randn(M)
        A = randn(rng, M, N)
        Λ = cholesky(A * A' + I)
        U = cholesky(pw(EQ(), z)).U
        cache = PPC(z, U, m̂ε, Λ)
    end

    @testset "ApproxCondMean" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x = collect(range(-3.0, 3.0; length=N))
        z = collect(range(-3.0, 3.0; length=M))
        m = ApproxCondMean(construct_cache(N, rng, z), CustomMean(sin), EQ())
        differentiable_mean_function_tests(rng, m, x)
    end
    @testset "ApproxCondKernel" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x0 = collect(range(-3.0, stop=3.0, length=N))
        x1 = collect(range(-3.5, stop=2.5, length=N))
        x2 = collect(range(-2.5, stop=3.5, length=N′))
        z = collect(range(-3.0, 3.0; length=M))

        k = ApproxCondKernel(construct_cache(N, rng, z), EQ(), EQ())
        differentiable_kernel_tests(rng, k, x0, x1, x2)
    end
    @testset "ApproxCondCrossKernel" begin
        rng, N, N′, M = MersenneTwister(123456), 11, 13, 7
        x0 = collect(range(-3.0, stop=3.0, length=N))
        x1 = collect(range(-3.5, stop=2.5, length=N))
        x2 = collect(range(-2.5, stop=3.5, length=N′))
        z = collect(range(-3.0, 3.0; length=M))

        k = ApproxCondCrossKernel(construct_cache(N, rng, z), EQ(), EQ(), EQ())
        differentiable_cross_kernel_tests(rng, k, x0, x1, x2)
    end
end
