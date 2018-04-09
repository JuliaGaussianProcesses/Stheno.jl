@testset "block_arrays" begin

    # Test construction of `BlockVector` from a vector of vectors.
    let
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x, x′ = randn(rng, N), randn(rng, N′)
        x̂ = BlockVector([x, x′])
        @test length(x̂) == N + N′
        @test x̂ == vcat(x, x′)
        @test getblock(x̂, 1) == x && getblock(x̂, 2) == x′
        @test transpose(getblock(x̂, 1)) == getblock(transpose(x̂), 1)
        @test adjoint(getblock(x̂, 1)) == getblock(adjoint(x̂), 1)
        @test blocksizes(x̂, 1) == [N, N′]
        @test blocklengths(x̂) == blocksizes(x̂, 1)
        @test blocklengths(x̂) == blocklengths(transpose(x̂))
        @test blocklengths(x̂) == blocklengths(adjoint(x̂))
    end

    # Test construction of `BlockMatrix` from matrix of matrices.
    let
        rng, P1, P2, P3, Q1, Q2 = MersenneTwister(123456), 2, 3, 6, 4, 5
        X11, X12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
        X21, X22 = randn(rng, P2, Q1), randn(rng, P2, Q2)
        X31, X32 = randn(rng, P3, Q1), randn(rng, P3, Q2)
        X = BlockMatrix(reshape([X11, X21, X31, X12, X22, X32], 3, 2))
        @test size(X, 1) == P1 + P2 + P3
        @test size(X, 2) == Q1 + Q2
        @test getblock(X, 1, 1) == X11
        @test getblock(X, 1, 2) == X12
        @test getblock(X, 2, 1) == X21
        @test getblock(X, 2, 2) == X22
        @test blocksizes(X, 1) == [P1, P2, P3]
        @test blocksizes(X, 2) == [Q1, Q2]
        @test blocksizes(X, 1) == blocksizes(X', 2)
        @test blocksizes(X, 2) == blocksizes(X', 1)
        @test blocksizes(X, 1) == blocksizes(transpose(X), 2)
        @test blocksizes(X, 2) == blocksizes(transpose(X), 1)
        @test BlockMatrix([X11, X21, X31, X12, X22, X32], 3, 2) == X
    end

    # Test multiplication of a `BlockMatrix` by a `BlockVector`.
    let
        rng, P1, P2, Q1, Q2 = MersenneTwister(123456), 2, 3, 4, 5
        X11, X12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
        X21, X22 = randn(rng, P2, Q1), randn(rng, P2, Q2)
        x1, x2 = randn(rng, Q1), randn(rng, Q2)
        A, x = BlockMatrix([X11, X21, X12, X22], 2, 2), BlockVector([x1, x2])
        @test length(A * x) == P1 + P2
        @test A * x ≈ Matrix(A) * Vector(x)

        x̂1, x̂2 = randn(rng, P1), randn(rng, P2)
        x̂ = BlockVector([x̂1, x̂2])
        @test A' * x̂ ≈ Matrix(A)' * Vector(x̂)
        @test transpose(A) * x̂ ≈ transpose(Matrix(A)) * x̂
    end

    # Test multiplication of a `BlockMatrix` by a `BlockMatrix`.
    let
        rng, P1, P2, Q1, Q2, R1, R2, R3 = MersenneTwister(123456), 2, 3, 4, 5, 6, 7, 8
        A11, A12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
        A21, A22 = randn(rng, P2, Q1), randn(rng, P2, Q2)
        B11, B12, B13 = randn(rng, Q1, R1), randn(rng, Q1, R2), randn(rng, Q1, R3)
        B21, B22, B23 = randn(rng, Q2, R1), randn(rng, Q2, R2), randn(rng, Q2, R3)
        A = BlockMatrix([A11, A21, A12, A22], 2, 2)
        B = BlockMatrix([B11, B21, B12, B22, B13, B23], 2, 3)
        @test size(A * B, 1) == P1 + P2
        @test size(A * B, 2) == R1 + R2 + R3
        @test A * B ≈ Matrix(A) * Matrix(B)
        @test B' * A' ≈ Matrix(B)' * Matrix(A)'
        @test transpose(B) * transpose(A) ≈ transpose(Matrix(B)) * transpose(Matrix(A))
    end

    # Test `chol`.
    let
        rng, P1, P2 = MersenneTwister(123456), 3, 4
        tmp = randn(rng, P1 + P2, P1 + P2)
        A_ = tmp * tmp' + 1e-9I
        A = BlockArray(A_, [P1, P2], [P1, P2])
        @assert A_ == A

        # Compute chols and compare.
        U_ = chol(Symmetric(A_))
        
    end
end
