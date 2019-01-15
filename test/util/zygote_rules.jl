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
    @test_throws ErrorException Zygote.forward(C->C.factors, C)

    # Test getproperty.
    adjoint_test(A->Cholesky(A, :U, 0).U, randn(rng, N, N), A)
    adjoint_test(A->Cholesky(A, :U, 0).L, randn(rng, N, N), A)
    adjoint_test(A->Cholesky(A, :L, 0).U, randn(rng, N, N), A)
    adjoint_test(A->Cholesky(A, :L, 0).L, randn(rng, N, N), A)
end

# Verify cholesky factorisation correctness.
let
    rng, N = MersenneTwister(123456), 3
    A = randn(rng, N, N)
    adjoint_test(A->logdet(cholesky(Symmetric(A' * A + 1e-3I))), randn(rng), A)
    adjoint_test(A->cholesky(Symmetric(A' * A + 1e-3I)).U, randn(rng, N, N), A)
    adjoint_test(A->cholesky(Symmetric(A' * A + 1e-3I)).L, randn(rng, N, N), A)

    adjoint_test(A->logdet(cholesky(A' * A + 1e-3I)), randn(rng), A)
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

end
