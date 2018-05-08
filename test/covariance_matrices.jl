@testset "covariance_matrices" begin

    let
        rng, N, P, Q = MersenneTwister(123456), 5, 6, 2
        B = randn(rng, N, N)
        A_ = B' * B + UniformScaling(1e-6)
        U = chol(A_)
        @test 2 * logdet(U) ≈ logdet(A_)
    end

    # Test that the AbstractArray interface works.
    let
        rng, N, P, Q = MersenneTwister(123456), 5, 6, 2
        B = randn(rng, N, N)
        Σ_ = B' * B + 1e-6I
        Σ = LazyPDMat(Σ_)
        @test size(Σ) == (N, N)
        @test getindex(Σ, 10) == getindex(Σ_, 10)
    end

    # Test strided matrix functionality.
    let
        # Set up some matrices.
        rng, N, P, Q = MersenneTwister(123456), 5, 6, 2
        B = randn(rng, N, N)
        A_ = B' * B + UniformScaling(1e-6)
        A = LazyPDMat(A_)
        x, X, Y = randn(rng, N), randn(rng, N, P), randn(rng, N, Q)

        # Check utility functionality.
        @test size(A) == size(A_)
        @test size(A, 1) == size(A_, 1)
        @test size(A, 2) == size(A_, 2)
        
        @test Matrix(A) == A_
        @test A == A

        # Test unary operations.
        @test logdet(A) ≈ logdet(A_)
        @test chol(A) == chol(A_ + A.ϵ * I)

        # Test binary operations.
        @test typeof(A + A) <: LazyPDMat
        @test Matrix(A + A) == A_ + A_
        @test typeof(A - A) <: LazyPDMat
        @test Matrix(A - A) == A_ - A_
        @test typeof(A * A) <: LazyPDMat
        @test Matrix(A * A) == A_ * A_
        @test typeof(map(*, A, A)) <: LazyPDMat
        @test map(*, A, A) == LazyPDMat(map(*, A_, A_))
        @test typeof(broadcast(*, A, A)) <: LazyPDMat
        @test broadcast(*, A, A) == LazyPDMat(A_ .* A_)

        # Specialised matrix operations.
        @test typeof(Xt_A_X(A, X)) <: LazyPDMat
        @test typeof(Xt_invA_X(A, X)) <: LazyPDMat
        @test X' * A_ * X ≈ Matrix(Xt_A_X(A, X))
        @test X' * A * Y ≈ Xt_A_Y(X, A, Y)
        @test X' * (A_ \ X) ≈ Matrix(Xt_invA_X(A, X))
        @test X' * (A_ \ Y) ≈ Xt_invA_Y(X, A, Y)
        @test A_ \ X ≈ A \ X
    end

    # Test misc. operations.
    let
        rng, N, N′, D = MersenneTwister(123456), 7, 8, 2
        A, B = randn(rng, D, N), randn(rng, D, N)
        @test Stheno.diagAᵀA(A) ≈ diag(A'A)
        @test Stheno.diagAᵀB(A, B) ≈ diag(A'B)
    end

end
