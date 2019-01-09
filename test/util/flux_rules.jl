using FDM, Flux, Distances, Random, LinearAlgebra

@testset "flux_rules" begin

# Check squared-euclidean distance implementation (AbstractMatrix)
let
    fdm = central_fdm(5, 1)
    rng, P, Q, D = MersenneTwister(123456), 10, 9, 8
    X, Y = randn(rng, D, P), randn(rng, D, Q)

    # Check first argument of binary pairwise.
    f = X->sum(pairwise(SqEuclidean(), X, Y))
    @test all(Flux.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument of binary pairwise.
    f = Y->sum(pairwise(SqEuclidean(), X, Y))
    @test all(Flux.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    # Check unary pairwise.
    @test Flux.gradient(X->sum(pairwise(SqEuclidean(), X)), X)[1] ≈
        Flux.gradient(X->sum(pairwise(SqEuclidean(), X, X)), X)[1]
end

# Check squared-euclidean distance implementation (AbstractVector)
let
    fdm = central_fdm(5, 1)
    rng, P, Q = MersenneTwister(123456), 10, 9
    x, y = randn(rng, P), randn(rng, Q)

    # Check first argument of binary pairwise.
    f = x->sum(pairwise(SqEuclidean(), x, y))
    @test all(Flux.gradient(f, x)[1] .- FDM.grad(fdm, f, x) .< 1e-8)

    # Check second argument of binary pairwise.
    f = y->sum(pairwise(SqEuclidean(), x, y))
    @test all(Flux.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

    # Check unary pairwise.
    @test Flux.gradient(x->sum(pairwise(SqEuclidean(), x)), x)[1] ≈
        Flux.gradient(x->sum(pairwise(SqEuclidean(), x, x)), x)[1]
end

# Check that \ and / work.
let
    fdm = central_fdm(5, 1)
    rng, P, Q = MersenneTwister(123456), 10, 9
    X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)

    # \

    # Check first argument sensitivity in matrix case.
    f = X->sum(X \ Y)
    @test all(Flux.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in matrix case.
    f = Y->sum(X \ Y)
    @test all(Flux.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    # Check first argument sensitivity in vector case.
    f = X->sum(X \ y)
    @test all(Flux.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in vector case.
    f = y->sum(X \ y)
    @test all(Flux.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

    # /

    # Check first argumeant sensitivity in matrix case.
    f = X->sum(Y' / X)
    @test all(Flux.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in matrix case.
    f = Y->sum(Y' / X)
    @test all(Flux.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    # Check first argument sensitivity in vector case.
    f = X->sum(y' / X)
    @test all(Flux.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in vector case.
    f = y->sum(y' / X)
    @test all(Flux.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)
end

# Check that Symmetric works as expected.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 7
    A = randn(rng, P, P)

    f = A->sum(Symmetric(A))
    @test all(Flux.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
end

# Check that `diag` behaves sensibly.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 10
    A = randn(rng, P, P)

    f = A->sum(diag(A))
    @test all(Flux.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
end

# Check that addition of matrices and uniform scalings works as hoped.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 10
    A = randn(rng, P, P)

    f = A->sum(A + 5I)
    @test all(Flux.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
end

end # testset
