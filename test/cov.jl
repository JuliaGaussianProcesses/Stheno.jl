@testset "cov.jl" begin
    let
        k = (x, y)->1.0
        P, Q = 5, 7
        K, x, y, X, Y = CovMat(P, Q), randn(P), randn(Q), randn(3, P), randn(3, Q)
        @test cov!(K, k, x, y) == ones(P, Q)
        @test cov!(K, k, x, y) == cov(k, x, y)
        @test cov!(K, k, X, Y) == ones(P, Q)
        @test cov!(K, k, X, Y) == cov(k, X, Y)
    end
end
