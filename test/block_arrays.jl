@testset "block_arrays" begin

    # Test construction of `BlockVector` from a vector of vectors. Also test copying.
    let
        rng, N, N′ = MersenneTwister(123456), 5, 6
        x, x′ = randn(rng, N), randn(rng, N′)
        x̂ = BlockVector([x, x′])
        @test length(x̂) == N + N′
        @test x̂ == vcat(x, x′)
        @test getblock(x̂, 1) == x && getblock(x̂, 2) == x′
        @test blocksizes(x̂, 1) == [N, N′]
        @test blocklengths(x̂) == blocksizes(x̂, 1)
        @test copy(x̂) == x̂
    end

    # Test construction of `BlockMatrix` from matrix of matrices. Also test copying.
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

        @test BlockMatrix([X11, X21, X31, X12, X22, X32], 3, 2) == X

        @test copy(X) == X
    end

    # Test transposition of block matrices.
    let
        rng, P1, P2, P3, Q1, Q2 = MersenneTwister(123456), 2, 3, 6, 4, 5
        X11, X12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
        X21, X22 = randn(rng, P2, Q1), randn(rng, P2, Q2)
        X31, X32 = randn(rng, P3, Q1), randn(rng, P3, Q2)
        X = BlockMatrix(reshape([X11, X21, X31, X12, X22, X32], 3, 2))

        for foo in [ctranspose, transpose]

            @test foo(X) isa AbstractBlockMatrix

            @test nblocks(foo(X), 1) == nblocks(X, 2)
            @test nblocks(foo(X), 2) == nblocks(X, 1)
            @test nblocks(foo(X)) == reverse(nblocks(X))

            @test blocksizes(foo(X), 1) == blocksizes(X, 2)
            @test blocksizes(foo(X), 2) == blocksizes(X, 1)

            @test size(foo(X), 1) == size(X, 2)
            @test size(foo(X), 2) == size(X, 1)
            @test size(foo(X)) == reverse(size(X))

            @test foo(foo(X)) == X
        end
    end

    # Test transposition of block vectors.
    let
        # @test transpose(getblock(x̂, 1)) == getblock(transpose(x̂), 1)
        # @test ctranspose(getblock(x̂, 1)) == getblock(ctranspose(x̂), 1)
        # @test blocklengths(x̂) == blocklengths(transpose(x̂))
        # @test blocklengths(x̂) == blocklengths(ctranspose(x̂))
    end

    # Test multiplication of a `BlockMatrix` by a `BlockVector`.
    let
        rng, P1, P2, Q1, Q2 = MersenneTwister(123456), 2, 3, 4, 5
        X11, X12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
        X21, X22 = randn(rng, P2, Q1), randn(rng, P2, Q2)
        x1, x2 = randn(rng, Q1), randn(rng, Q2)
        A, x = BlockMatrix([X11, X21, X12, X22], 2, 2), BlockVector([x1, x2])
        @test length(A * x) == P1 + P2
        @test A * x isa AbstractBlockVector
        @test A * x ≈ Matrix(A) * Vector(x)

        x̂1, x̂2 = randn(rng, P1), randn(rng, P2)
        x̂ = BlockVector([x̂1, x̂2])

        for foo in [ctranspose, transpose]
            @test foo(A) isa AbstractBlockMatrix
            @test foo(A) * x̂ isa AbstractBlockVector
            @test foo(A) * x̂ ≈ Matrix(A)' * Vector(x̂)
            @test foo(A) * x̂ ≈ foo(Matrix(A)) * x̂
        end
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

        for foo in [ctranspose, transpose]
            @test foo(B) * foo(A) ≈ foo(Matrix(B)) * foo(Matrix(A))
        end
    end

    # Test SquareDiagonal matrix construction and util.
    let
        rng, P1, P2 = MersenneTwister(123456), 3, 4
        A11, A12, A22 = randn(rng, P1, P1), randn(rng, P1, P2), randn(rng, P2, P2)
        A_ = BlockMatrix([A11, zeros(Float64, P2, P1), A12, A22], 2, 2)
        A = SquareDiagonal(A_)
        @test nblocks(A) == nblocks(A_)
        @test nblocks(A, 1) == nblocks(A_, 1)
        @test nblocks(A, 2) == nblocks(A_, 2)
        @test blocksize(A, 1, 1) == blocksize(A_, 1, 1) &&
            blocksize(A, 1, 2) == blocksize(A_, 1, 2) &&
            blocksize(A, 2, 1) == blocksize(A_, 2, 1) &&
            blocksize(A, 2, 2) == blocksize(A_, 2, 2)
        @test size(A) == size(A_)
        @test getindex(A, 1, 2) == getindex(A_, 1, 2)
        @test eltype(A) == eltype(A_)
        @test copy(A) == A

        # UpperTriangular functionality.
        @test getblock(UpperTriangular(A), 1, 1) == getblock(A, 1, 1)
        @test getblock(UpperTriangular(A), 2, 1) == zeros(P2, P1)
        @test getblock(UpperTriangular(A), 1, 2) == getblock(A, 1, 2)

        # LowerTriangular functionality.
        @test getblock(LowerTriangular(A), 1, 1) == getblock(A, 1, 1)
        @test getblock(LowerTriangular(A), 1, 2) == zeros(P1, P2)
        @test getblock(LowerTriangular(A), 2, 1) == getblock(A, 2, 1)
        @test getblock(LowerTriangular(A), 2, 2) == getblock(A, 2, 2)
    end

    # Test `chol`, logdet, and backsolving.
    let
        rng, P1, P2 = MersenneTwister(123456), 3, 4
        tmp = randn(rng, P1 + P2, P1 + P2)
        A_ = tmp * tmp' + 1e-9I
        A = BlockArray(A_, [P1, P2], [P1, P2])
        @assert A_ == A

        # Compute chols and compare.
        U_, U = chol(Symmetric(A_)), chol(SquareDiagonal(A))
        @test U_ ≈ U

        # Test `logdet`.
        @test logdet(U) ≈ logdet(U_)

        # Test backsolving for block vector.
        x1, x2 = randn(rng, P1), randn(rng, P2)
        x = BlockVector([x1, x2])

        @test typeof(U \ x) <: AbstractBlockVector
        @test size(U \ x) == size(U_ \ Vector(x))
        @test U \ x ≈ U_ \ Vector(x)

        @test typeof(U') <: LowerTriangular{<:Real, <:AbstractBlockMatrix}
        @test typeof(U' \ x) <: AbstractBlockVector
        @test typeof(Ac_ldiv_B(U, x)) <: AbstractBlockVector
        @test size(U' \ x) == size(U_' \ Vector(x))
        @test U' \ x ≈ U_' \ Vector(x)

        @test typeof(transpose(U) \ x) <: AbstractBlockVector
        @test size(transpose(U) \ x) == size(U_' \ Vector(x))
        @test transpose(U) \ x ≈ U_' \ Vector(x)

        # Test backsolving for block matrix
        Q1, Q2 = 7, 6
        X11, X12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
        X21, X22 = randn(rng, P2, Q1), randn(rng, P2, Q2) 
        X = BlockMatrix([X11, X21, X12, X22], 2, 2)

        @test typeof(U \ X) <: AbstractBlockMatrix
        @test size(U \ X) == size(U_ \ Matrix(X))
        @test U \ X ≈ U_ \ Matrix(X)

        @test typeof(U' \ X) <: AbstractBlockMatrix
        @test size(U' \ X) == size(U_' \ Matrix(X))
        @test U' \ X ≈ U_' \ Matrix(X)
    end

    # Test UniformScaling interaction.
    let
        rng, N = MersenneTwister(123456), 5
        X = SquareDiagonal(BlockArray(randn(rng, N, N), [2, 3], [2, 3]))
        @test X + I == Matrix(X) + I
        @test I + X == I + Matrix(X)
    end

    # Run covariance matrix tests with a BlockMatrix.
    let
        rng, P1, P2, Q1, Q2, R1, R2 = MersenneTwister(123456), 2, 3, 4, 5, 6, 7
        P, Q, R = P1 + P2, Q1 + Q2, R1 + R2
        B = randn(rng, P, P)
        A_ = BlockArray(B' * B + UniformScaling(1e-6), [P1, P2], [P1, P2])
        A = LazyPDMat(SquareDiagonal(A_))
        x = BlockArray(randn(rng, P), [P1, P2])
        X = BlockArray(randn(rng, P, Q), [P1, P2], [Q1, Q2])
        X′ = BlockArray(randn(rng, P, R), [P1, P2], [R1, R2])

        # Check utility functionality.
        @test size(A) == size(A_)
        @test size(A, 1) == size(A_, 1)
        @test size(A, 2) == size(A_, 2)

        @test Matrix(A) == A_
        @test A == A

        # Test unary operations.
        @test logdet(A) ≈ logdet(Matrix(A_))
        @test chol(A) ≈ chol(Matrix(A_) + A.ϵ * I)

        # Test binary operations.
        @test typeof(A_ + A_) <: AbstractBlockMatrix
        @test Matrix(A + A) == A_ + A_
        @test typeof(A_ - A_) <: AbstractBlockMatrix
        @test Matrix(A - A) == A_ - A_
        # @test Matrix(A * A) == A_ * A_
        @test typeof(chol(A).data) <: AbstractBlockMatrix
        @test x' * (Matrix(A_) \ x) ≈ Xt_invA_X(A, x)
        @test typeof(Xt_invA_X(A, X)) <: LazyPDMat
        @test typeof(chol(Xt_invA_X(A, X)).data) <: AbstractBlockMatrix
        @test maximum(abs.(X' * (Matrix(A_) \ X) - Matrix(Xt_invA_X(A, X)))) < 1e-6
        @test typeof(Xt_invA_Y(X′, A, X)) <: AbstractBlockMatrix
        @test maximum(abs.(X′' * (Matrix(A_) \ X) - Xt_invA_Y(X′, A, X))) < 1e-6
        @test typeof(A \ X) <: AbstractBlockMatrix
        @test maximum(abs.(Matrix(A_) \ X - A \ X)) < 1e-6
    end
end
