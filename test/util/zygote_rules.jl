using FDM, Zygote, Distances, Random, LinearAlgebra
using Stheno: chol

@testset "zygote_rules" begin

# Check squared-euclidean distance implementation (AbstractMatrix)
let
    fdm = central_fdm(5, 1)
    rng, P, Q, D = MersenneTwister(123456), 10, 9, 8
    X, Y = randn(rng, D, P), randn(rng, D, Q)

    # Check first argument of binary pairwise.
    f = X->sum(pairwise(SqEuclidean(), X, Y))
    @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument of binary pairwise.
    f = Y->sum(pairwise(SqEuclidean(), X, Y))
    @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    # Check unary pairwise.
    @test Zygote.gradient(X->sum(pairwise(SqEuclidean(), X)), X)[1] ≈
        Zygote.gradient(X->sum(pairwise(SqEuclidean(), X, X)), X)[1]
end

# Check squared-euclidean distance implementation (AbstractVector)
let
    fdm = central_fdm(5, 1)
    rng, P, Q = MersenneTwister(123456), 10, 9
    x, y = randn(rng, P), randn(rng, Q)

    # Check first argument of binary pairwise.
    f = x->sum(pairwise(SqEuclidean(), x, y))
    @test all(Zygote.gradient(f, x)[1] .- FDM.grad(fdm, f, x) .< 1e-8)

    # Check second argument of binary pairwise.
    f = y->sum(pairwise(SqEuclidean(), x, y))
    @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

    # Check unary pairwise.
    @test Zygote.gradient(x->sum(pairwise(SqEuclidean(), x)), x)[1] ≈
        Zygote.gradient(x->sum(pairwise(SqEuclidean(), x, x)), x)[1]
end

# Check that \ and / work.
let
    fdm = central_fdm(5, 1)
    rng, P, Q = MersenneTwister(123456), 10, 9
    X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)

    # \

    # Check first argument sensitivity in matrix case.
    f = X->sum(X \ Y)
    @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in matrix case.
    f = Y->sum(X \ Y)
    @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    # Check first argument sensitivity in vector case.
    f = X->sum(X \ y)
    @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in vector case.
    f = y->sum(X \ y)
    @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)

    # /

    # Check first argumeant sensitivity in matrix case.
    f = X->sum(Y' / X)
    @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in matrix case.
    f = Y->sum(Y' / X)
    @test all(Zygote.gradient(f, Y)[1] .- FDM.grad(fdm, f, Y) .< 1e-8)

    # Check first argument sensitivity in vector case.
    f = X->sum(y' / X)
    @test all(Zygote.gradient(f, X)[1] .- FDM.grad(fdm, f, X) .< 1e-8)

    # Check second argument sensitivity in vector case.
    f = y->sum(y' / X)
    @test all(Zygote.gradient(f, y)[1] .- FDM.grad(fdm, f, y) .< 1e-8)
end

# Check that Symmetric works as expected.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 7
    A = randn(rng, P, P)

    f = A->sum(Symmetric(A))
    @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
end

# Check that cholesky works as expected.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 7
    A = randn(rng, P, P)
    Σ = A'A + 1e-6I

    f = Σ->sum(chol(Symmetric(Σ)))
    @test all(Zygote.gradient(f, Σ)[1] .- FDM.grad(fdm, f, Σ) .< 1e-8)
end

# Check that `diag` behaves sensibly.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 10
    A = randn(rng, P, P)

    f = A->sum(diag(A))
    @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
end

# Check that `logdet` works as expected for UpperTriangular matrices.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 10
    A = randn(rng, P, P)
    Σ = A'A + 1e-6I

    f = Σ->logdet(chol(Symmetric(Σ)))
    @test all(Zygote.gradient(f, Σ)[1] .- FDM.grad(fdm, f, Σ) .< 1e-8)
end

# Check that addition of matrices and uniform scalings works as hoped.
let
    fdm = central_fdm(5, 1)
    rng, P = MersenneTwister(123456), 10
    A = randn(rng, P, P)

    f = A->sum(A + 5I)
    @test all(Zygote.gradient(f, A)[1] .- FDM.grad(fdm, f, A) .< 1e-8)
end

end
