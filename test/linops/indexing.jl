using Random, LinearAlgebra
using Stheno: GPC

@testset "indexing" begin
    @testset "no noise" begin
        f, x = GP(eq(), GPC()), randn(10)
        fx = f(x)
        @test mean(fx) == map(mean(f), x)
        @test cov(fx) == pairwise(kernel(f), x)
    end
    @testset "matrix-valued noise" begin
        rng, N = MersenneTwister(123456), 10
        f, x, A = GP(eq(), GPC()), randn(rng, N), randn(rng, N, N)
        C = Symmetric(A * A' + I)
        fx = f(x, C)
        @test mean(fx) == map(mean(f), x)
        @test cov(fx) == pairwise(kernel(f), x) + C
    end
end
