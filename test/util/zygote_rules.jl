using FDM, Zygote, Distances, Random, LinearAlgebra, FillArrays, ToeplitzMatrices

@testset "zygote_rules" begin

# Check FillArrays work as expected.
let
    rng, N = MersenneTwister(123456), 10
    @test Zygote.gradient(x->sum(Fill(x, N)), randn(rng))[1] == N
    @test Zygote.gradient(x->sum(Fill(x, N, 3, 4)), randn(rng))[1] == N * 3 * 4

    # Test unary broadcasting gradients.
    x = randn(rng)
    out, back = Zygote.forward(x->exp.(x), Fill(x, N))
    @test out isa Fill
    @test out == Fill(exp(x), N)
    @test back(Ones(N))[1] isa Fill
    @test back(Ones(N))[1] == Ones(N) .* exp(x)
    @test back(ones(N))[1] isa Vector
    @test back(ones(N))[1] == ones(N) .* exp(x)
    adjoint_test(x->exp.(x), randn(rng, N), Fill(x, N))
end

# Check squared-euclidean distance implementation (AbstractMatrix)
let
    rng, P, Q, D = MersenneTwister(123456), 10, 9, 8

    # Check sqeuclidean.
    let
        x, y = randn(rng, D), randn(rng, D)
        adjoint_test(x->sqeuclidean(x, y), randn(rng), x)
        adjoint_test(y->sqeuclidean(x, y), randn(rng), y)
    end

    # Check binary colwise.
    let
        X, Y = randn(rng, D, P), randn(rng, D, P)
        adjoint_test(X->colwise(SqEuclidean(), X, Y), randn(rng, P), X)
        adjoint_test(Y->colwise(SqEuclidean(), X, Y), randn(rng, P), Y)
    end

    # Check binary pairwise.
    let
        X, Y = randn(rng, D, P), randn(rng, D, Q)
        adjoint_test(X->pairwise(SqEuclidean(), X, Y), randn(rng, P, Q), X)
        adjoint_test(Y->pairwise(SqEuclidean(), X, Y), randn(rng, P, Q), Y)
    end

    # Check unary pairwise.
    let
        X = randn(rng, D, P)
        adjoint_test(X->pairwise(SqEuclidean(), X), randn(rng, P, P), X)
    end
end

# Check that \ and / work.
let
    rng, P, Q = MersenneTwister(123456), 10, 9
    X, Y, y = randn(rng, P, P), randn(rng, P, Q), randn(rng, P)

    # \
    adjoint_test(X->X \ Y, randn(rng, size(Y)), X)
    adjoint_test(Y->X \ Y, randn(rng, size(Y)), Y)
    adjoint_test(X->X \ y, randn(rng, size(y)), X)
    adjoint_test(y->X \ y, randn(rng, size(y)), y)

    # /
    Y′, y′ = collect(Y'), collect(y')
    adjoint_test(X->Y′ / X, randn(rng, size(Y′)), X)
    adjoint_test(Y′->Y′ / X, randn(rng, size(Y′)), Y′)
    adjoint_test(X->y′ / X, randn(rng, size(y′)), X)
    adjoint_test(y′->y′ / X, randn(rng, size(y′)), y′)
end

# Check that Symmetric works as expected.
let
    rng, P = MersenneTwister(123456), 7
    adjoint_test(Symmetric, randn(rng, P, P), randn(rng, P, P))
end

# Check that `diag` behaves sensibly.
let
    rng, P = MersenneTwister(123456), 10
    adjoint_test(diag, randn(rng, P), randn(rng, P, P))
end

# Verify that Cholesky retrieves gradient information correctly.
let
    rng, N = MersenneTwister(123456), 2
    A, uplo, info = randn(rng, N, N), :U, 0
    _, back = Zygote.forward(Cholesky, A, uplo, info)
    @test back((factors=A, uplo=nothing, info=nothing)) == (A, nothing, nothing)
end

# Verify Cholesky factorisation `getproperty`.
let
    rng, N = MersenneTwister(123456), 5
    A = randn(rng, N, N)
    S = A' * A + 1e-6I
    C = cholesky(S)

    # Check that non-differentiable ops run forwards and have `nothing` gradients.
    _, back = Zygote.forward(C->C.info, C)
    @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)
    _, back = Zygote.forward(C->C.uplo, C)
    @test back(1)[1] == (uplo=nothing, info=nothing, factors=nothing)

    # Unit test retrieving the factors.
    @test_broken Zygote.gradient(C->sum(C.factors), C)[1].factors == ones(size(C))

    # Integration testing with backprop and comparison with FDM.
    let
        f = A->sum(cholesky(A' * A + 1e-6I).factors)
        Ā_fd = FDM.grad(central_fdm(5, 1), f, A)
        @test_broken all(abs.(Zygote.gradient(f, A)[1] .- Ā_fd) .< 1e-8)
    end

    # Test getproperty.
    adjoint_test(A->Cholesky(A, :U, 0).U, randn(rng, N, N), A)
    adjoint_test(A->Cholesky(A, :U, 0).L, randn(rng, N, N), A)
    adjoint_test(A->Cholesky(A, :L, 0).U, randn(rng, N, N), A)
    adjoint_test(A->Cholesky(A, :L, 0).L, randn(rng, N, N), A)
end

# Verify cholesky factorisation correctness.
let
    rng, N = MersenneTwister(123456), 5
    A = randn(rng, N, N)
    adjoint_test(A->logdet(cholesky(Symmetric(A' * A + 1e-3I))), randn(rng), A)
    adjoint_test(A->cholesky(Symmetric(A' * A + 1e-3I)).U, randn(rng, N, N), A)
    adjoint_test(A->cholesky(Symmetric(A' * A + 1e-3I)).L, randn(rng, N, N), A)
    adjoint_test(A->cholesky(A' * A + 1e-3I).U, randn(rng, N, N), A)
    adjoint_test(A->cholesky(A' * A + 1e-3I).L, randn(rng, N, N), A)
end

# Verify Cholesky factorisation of Toeplitz matrices correctness.
let
    rng, N = MersenneTwister(123456), 10
    x = pairwise(EQ(), range(-3.0, stop=3.0, length=N))[:, 1]
    x[1] += 0.1 # ensure positive definite-ness under minor perturbations.
    adjoint_test(
        x->cholesky(SymmetricToeplitz(x)).U,
        randn(rng, N, N),
        x;
        rtol=1e-6,
        atol=1e-6,
    )
end

# Check that addition of matrices and uniform scalings works as hoped.
let
    rng, N = MersenneTwister(123456), 10
    A, λ = randn(rng, N, N), randn(rng)
    adjoint_test(A->A + λ * I, randn(rng, N, N), A)
    adjoint_test(λ->A + λ * I, randn(rng, N, N), λ)
end

end
