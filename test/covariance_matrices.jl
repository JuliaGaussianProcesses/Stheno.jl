@testset "strided_covmat" begin

    let
        rng, N, P, Q = MersenneTwister(123456), 5, 6, 2
        B = randn(rng, N, N)
        A_ = B' * B + UniformScaling(1e-6)
        U = chol(A_)
        @test 2 * logdet(U) ≈ logdet(A_)
    end

    # Test strided matrix functionality.
    let
        # Set up some matrices.
        rng, N, P, Q = MersenneTwister(123456), 5, 6, 2
        B = randn(rng, N, N)
        A_ = B' * B + UniformScaling(1e-6)
        A = LazyPDMat(A_)
        x, X, X′ = randn(rng, N), randn(rng, N, P), randn(rng, N, Q)

        # Check utility functionality.
        @test size(A) == size(A_)
        @test size(A, 1) == size(A_, 1)
        @test size(A, 2) == size(A_, 2)
        
        @test Matrix(A) == A_
        @test A == A

        # Test unary operations.
        @test logdet(A) ≈ logdet(A_)
        @test chol(A) == chol(A_ + Stheno.__ϵ * I)

        # Test binary operations.
        @test Matrix(A + A) == A_ + A_
        @test Matrix(A - A) == A_ - A_
        @test Matrix(A * A) == A_ * A_
        @test x' * (A_ \ x) ≈ invquad(A, x)
        @test typeof(Xt_invA_X(A, X)) <: LazyPDMat
        @test X' * (A_ \ X) ≈ Matrix(Xt_invA_X(A, X))
        @test X' * (A_ \ X′) ≈ Xt_invA_Y(X, A, X′)
        @test A_ \ X ≈ A \ X
    end
end
