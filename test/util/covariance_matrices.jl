@timedtestset "cholesky" begin
    # Set up some matrices and factorisations.
    rng, N, N′, P, Q = MersenneTwister(123456), 5, 3, 6, 2
    B = randn(rng, N, N)
    A_ = B' * B + UniformScaling(1e-6)
    A = cholesky(A_)
    x, y, z = randn(rng, N), randn(rng, N′), randn(rng, N)
    X, Y = randn(rng, N, P), randn(rng, N, Q)
    Z = randn(rng, N, P)
    A_1_ = exp(randn(rng))
    A_1 = cholesky(A_1_)

    # Specialised matrix operations.
    @test Xt_invA_X(A, x) isa Real
    @test Xt_invA_X(A, x) ≈ x' * (A \ x)

    @test Xt_invA_X(A, X) isa Symmetric
    @test Xt_invA_X(A, X) ≈ X' * (A \ X)

    @test Xt_invA_Y(X, A, Y) isa Matrix
    @test Xt_invA_Y(X, A, Y) ≈ X' * (A \ Y)

    @test diag_At_A(x) ≈ [x'x]
    @test diag_At_A(X) ≈ diag(X'X)

    @test diag_At_B(x, z) ≈ [x'z]
    @test diag_At_B(X, Z) ≈ diag(X'Z)

    @test diag_Xt_invA_X(A, X) ≈ diag(Xt_invA_X(A, X))

    @test diag_Xt_invA_Y(X, A, Z) ≈ diag(Xt_invA_Y(X, A, Z))
end
