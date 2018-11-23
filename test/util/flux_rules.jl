using FDM, Flux, Distances

@testset "flux_rules" begin

# Check squared-euclidean distance implementation.
let
    fdm = central_fdm(5, 1)
    rng, P, Q, D = MersenneTwister(123456), 10, 9, 8
    X, Y = randn(rng, D, P), randn(rng, D, Q)

    f = X->sum(pairwise(SqEuclidean(), X, Y))
    @test all(Flux.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    f = Y->sum(pairwise(SqEuclidean(), X, Y))
    @test all(Flux.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    @test Flux.gradient(X->sum(pairwise(SqEuclidean(), X)), X)[1] ≈
        Flux.gradient(X->sum(pairwise(SqEuclidean(), X, X)), X)[1]
end

end # testset
