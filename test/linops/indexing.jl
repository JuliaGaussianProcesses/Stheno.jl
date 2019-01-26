using Random, LinearAlgebra
using Stheno: GPC

@testset "indexing" begin
    f, x = GP(EQ(), GPC()), randn(10)
    fx = f(x)
    @test mean(fx) == map(mean(f), x)
    @test cov(fx) == pairwise(kernel(f), x)
end # @testset indexing
