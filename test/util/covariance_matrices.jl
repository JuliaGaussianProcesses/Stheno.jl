using Random, LinearAlgebra
using Stheno: Xt_A_X, Xt_A_Y, Xt_invA_Y, Xt_invA_X, diag_At_A, diag_At_B, diag_Xt_A_X,
    diag_Xt_A_Y, diag_Xt_invA_X, diag_Xt_invA_Y, Xtinv_A_Xinv


@testset "cholesky" begin

    # Test additional operations for Cholesky factorisations.
    let
        # Set up some matrices and factorisations.
        rng, N, P, Q = MersenneTwister(123456), 5, 6, 2
        B = randn(rng, N, N)
        A_ = B' * B + UniformScaling(1e-6)
        A = cholesky(A_)
        x, X, Y = randn(rng, N), randn(rng, N, P), randn(rng, N, Q)
        Z = randn(rng, N, P)

        # Specialised matrix operations.

        @test Xt_A_X(A, x) isa Real
        @test Xt_A_X(A, x) ≈ x' * A_ * x

        @test Xt_A_X(A, X) isa Symmetric
        @test Xt_A_X(A, X) ≈ X' * A_ * X

        @test Xt_A_Y(X, A, Y) isa Matrix
        @test Xt_A_Y(X, A, Y) ≈ X' * A_ * Y

        @test Xt_invA_X(A, x) isa Real
        @test Xt_invA_X(A, x) ≈ x' * (A \ x)

        @test Xt_invA_X(A, X) isa Symmetric
        @test Xt_invA_X(A, X) ≈ X' * (A \ X)

        @test Xt_invA_Y(X, A, Y) isa Matrix
        @test Xt_invA_Y(X, A, Y) ≈ X' * (A \ Y)

        @test diag_At_A(X) ≈ diag(X'X)

        @test diag_At_B(X, Z) ≈ diag(X'Z)

        @test diag_Xt_A_X(A, X) ≈ diag(Xt_A_X(A, X))

        @test diag_Xt_A_Y(X, A, Z) ≈ diag(Xt_A_Y(X, A, Z))

        @test diag_Xt_invA_X(A, X) ≈ diag(Xt_invA_X(A, X))

        @test diag_Xt_invA_Y(X, A, Z) ≈ diag(Xt_invA_Y(X, A, Z))

        @test Xtinv_A_Xinv(A, A) isa Symmetric
        @test Xtinv_A_Xinv(A, A) ≈ A \ (A \ A_)'
    end
end
