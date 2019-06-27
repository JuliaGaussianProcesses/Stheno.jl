using Random, LinearAlgebra, BlockArrays
using BlockArrays: cumulsizes, _BlockArray, BlockSizes

@testset "dense" begin

    # Test construction of a BlockArray via _BlockArray
    @testset "_BlockArray" begin
        rng, P, Q, R = MersenneTwister(123456), 7, 13, 11
        @testset "Block Vector" begin
            x, blk_sizes = randn(rng, P), BlockSizes([3, 2, 2])
            ȳ = randn(rng, P)
            x_blks = BlockArray(x, blk_sizes).blocks
            ȳ_blks = BlockArray(ȳ, blk_sizes).blocks

            y, back = Zygote.forward(_BlockArray, x_blks, blk_sizes)
            @test y == _BlockArray(x_blks, blk_sizes)
            @test first(back(ȳ)) == ȳ_blks
            @test last(back(ȳ)) === nothing
            @test first(back(ȳ)) == first(back(BlockArray(ȳ, blk_sizes)))
            @test first(back(ȳ)) == first(back((blocks=ȳ_blks, block_sizes=nothing)))
        end
        @testset "Block Array 3" begin
            X, blk_sizes = randn(rng, P, Q, R), BlockSizes([3, 4], [7, 6], [2, 3, 6])
            Ȳ = randn(rng, P, Q, R)
            X_blks = BlockArray(X, blk_sizes).blocks
            Ȳ_blks = BlockArray(Ȳ, blk_sizes).blocks

            Y, back = Zygote.forward(_BlockArray, X_blks, blk_sizes)
            @test Y == _BlockArray(X_blks, blk_sizes)
            @test first(back(Ȳ)) == Ȳ_blks
            @test last(back(Ȳ)) === nothing
            @test first(back(Ȳ)) == first(back(BlockArray(Ȳ, blk_sizes)))
            @test first(back(Ȳ)) == first(back((blocks=Ȳ_blks, block_sizes=nothing)))
        end
    end

    @testset "Vector(::BlockVector)" begin
        rng, Ps = MersenneTwister(123456), [5, 6, 7]
        x = BlockArray(randn(rng, sum(Ps)), Ps)
        adjoint_test(Vector, randn(rng, sum(Ps)), x)
    end

    @testset "Matrix(::BlockMatrix)" begin
        rng, Ps, Qs = MersenneTwister(123456), [3, 4, 5], [6, 7, 8, 9]
        X = BlockArray(randn(rng, sum(Ps), sum(Qs)), Ps, Qs)
        adjoint_test(Matrix, randn(rng, sum(Ps), sum(Qs)), X)
    end

    # # Test construction of `BlockVector` from a vector of vectors. Also test copying.
    # @testset "BlockVector" begin
    #     rng, N, N′ = MersenneTwister(123456), 5, 6
    #     x, x′ = randn(rng, N), randn(rng, N′)
    #     x̂ = BlockVector([x, x′])
    #     @test length(x̂) == N + N′
    #     @test x̂ == vcat(x, x′)
    #     @test getblock(x̂, 1) == x && getblock(x̂, 2) == x′

    #     # Test BlockVector construction adjoint.
    #     adjoint_test(x->BlockVector([x]), BlockVector([randn(rng, N)]), x)
    #     adjoint_test(x->BlockVector([x]), randn(rng, N), x)

    #     adjoint_test(
    #         x->BlockVector([x, x′]),
    #         BlockVector([randn(rng, N), randn(rng, N′)]),
    #         x,
    #     )
    #     adjoint_test(x->BlockVector([x, x′]), randn(rng, N + N′), x)

    #     # Test BlockVector conversion to Vector adjoint.
    #     adjoint_test(Vector, randn(rng, N), BlockVector([x]))
    #     adjoint_test(Vector, randn(rng, N + N′), x̂)

    #     # zero of a BlockVector should be a BlockVector
    #     @test zero(x̂) isa BlockVector
    #     @test cumulsizes(x̂) == cumulsizes(zero(x̂))
    #     @test zero(x̂) == zero(Vector(x̂))
    # end

    # # Test construction of `BlockMatrix` from matrix of matrices. Also test copying.
    # @testset "BlockMatrix" begin
    #     rng, P1, P2, P3, Q1, Q2 = MersenneTwister(123456), 2, 3, 6, 4, 5
    #     X11, X12 = randn(rng, P1, Q1), randn(rng, P1, Q2)
    #     X21, X22 = randn(rng, P2, Q1), randn(rng, P2, Q2)
    #     X31, X32 = randn(rng, P3, Q1), randn(rng, P3, Q2)
    #     X = BlockMatrix(reshape([X11, X21, X31, X12, X22, X32], 3, 2))
    #     @test size(X, 1) == P1 + P2 + P3
    #     @test size(X, 2) == Q1 + Q2
    #     @test getblock(X, 1, 1) == X11
    #     @test getblock(X, 1, 2) == X12
    #     @test getblock(X, 2, 1) == X21
    #     @test getblock(X, 2, 2) == X22

    #     @test cumulsizes(X, 1) == cumsum([1, P1, P2, P3])
    #     @test cumulsizes(X, 2) == cumsum([1, Q1, Q2])

    #     @test blocksizes(X, 1) == [P1, P2, P3]
    #     @test blocksizes(X, 2) == [Q1, Q2]

    #     @test BlockMatrix([X11, X21, X31, X12, X22, X32], 3, 2) == X

    #     Y = BlockMatrix([X11, X21, X31])
    #     @test size(Y, 1) == P1 + P2 + P3
    #     @test size(Y, 2) == Q1
    #     @test getblock(Y, 1, 1) == X11
    #     @test getblock(Y, 2, 1) == X21
    #     @test cumulsizes(Y, 1) == cumsum([1, P1, P2, P3])
    #     @test cumulsizes(Y, 2) == cumsum([1, Q1])

    #     @test blocksizes(Y, 1) == [P1, P2, P3]
    #     @test blocksizes(Y, 2) == [Q1]

    #     P, Q = P1 + P2 + P3, Q1 + Q2
    #     adjoint_test(
    #         X11->BlockMatrix(reshape([X11, X21, X31, X12, X22, X32], 3, 2)),
    #         randn(rng, P, Q),
    #         X11,
    #     )
    #     adjoint_test(
    #         X11->BlockMatrix(reshape([X11, X21, X31, X12, X22, X32], 3, 2)),
    #         X,
    #         X11,
    #     )
    #     adjoint_test(X11->BlockMatrix([X11, X21]), randn(rng, P1 + P2, Q1), X11)
    #     adjoint_test(X11->BlockMatrix([X11, X21]), BlockMatrix([X11, X21]), X11)
    #     adjoint_test(X11->BlockMatrix([X11, X12], 1, 2), BlockMatrix([X11, X12], 1, 2), X11)

    #     # BlockMatrix conversion to Matrix.
    #     adjoint_test(Matrix, randn(rng, size(X)), X)

    #     # zero of a BlockMatrix should be a BlockMatrix.
    #     @test zero(X) isa BlockMatrix
    #     @test cumulsizes(X) == cumulsizes(zero(X))
    #     @test zero(X) == zero(Matrix(X))
    # end

    # @testset "dense, transpose, and adjoint mul" begin
    #     rng, Ps, Qs, Rs = MersenneTwister(123456), [2, 3], [4, 5], [6, 7]
    #     X = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

    #     for foo in [transpose, adjoint]
    #         @test foo(X) isa BlockMatrix
    #         @test blocksizes(foo(X), 1) == blocksizes(X, 2)
    #         @test blocksizes(foo(X), 2) == blocksizes(X, 1)
    #         @test foo(Matrix(X)) == Matrix(foo(X))
    #         @test foo(foo(X)) == X
    #     end

    #     y = _BlockArray([randn(rng, Q) for Q in Qs], Qs)
    #     z = _BlockArray([randn(rng, P) for P in Ps], Ps)
    #     dense_BlockMatrix_BlockVector_mul_tests(rng, X, y)
    #     dense_BlockMatrix_BlockVector_mul_tests(rng, X', z)

    #     Y = _BlockArray([randn(rng, Q, R) for Q in Qs, R in Rs], Qs, Rs)

    #     Xt_blocks = Matrix.(transpose(X.blocks))
    #     Xt = _BlockArray(Xt_blocks, Qs, Ps)
    #     Yt_blocks = Matrix.(transpose(Y.blocks))
    #     Yt = _BlockArray(Yt_blocks, Rs, Qs)

    #     dense_BlockMatrix_BlockMatrix_mul_tests(rng, X, Y)
    #     for foo in [transpose, adjoint]
    #         dense_BlockMatrix_BlockMatrix_mul_tests(rng, foo(Xt), Y)
    #         dense_BlockMatrix_BlockMatrix_mul_tests(rng, X, foo(Yt))
    #         dense_BlockMatrix_BlockMatrix_mul_tests(rng, foo(Xt), foo(Yt))
    #     end
    # end

    # @testset "dense cholesky" begin
    #     rng, Ps = MersenneTwister(123456), [3, 4, 5]
    #     tmp = randn(rng, sum(Ps), sum(Ps))
    #     A_ = tmp * tmp' + 1e-9I
    #     A = BlockArray(A_, Ps, Ps)
    #     @assert A_ == A

    #     # Compute cholesky and compare.
    #     C_, C = cholesky(A_), cholesky(A)
    #     @test C isa Cholesky
    #     @test C_.U ≈ Matrix(C.U)
    #     @test C.U isa UpperTriangular{T, <:AbstractBlockMatrix{T}} where T

    #     function cholesky_gradients(C̄_factors, A_, A)

    #         # Compute cholesky adjoint with Matrix.
    #         C̄_dense = (factors=C̄_factors, uplo=nothing, info=nothing)
    #         _, back_dense = Zygote.forward(cholesky, A_)
    #         Ā_ = first(back_dense(C̄))

    #         # Compute cholesky adjoint with BlockMatrix.
    #         C̄_block = (factors=BlockArray(C̄_factors, blocksizes(A)...),)
    #         _, back_block = Zygote.forward(cholesky, A)
    #         Ā = first(back_block(C̄_block))

    #         # Ensure approximate agreement.
    #         @test Ā_ ≈ Ā
    #     end

    #     cholesky_gradients(randn(rng, size(A)), A_, A)
    #     cholesky_gradients(UpperTriangular(randn(rng, size(A))), A_, A)

    #     # Test `logdet`.
    #     @test logdet(C) ≈ logdet(C_)

    #     # Check that the forwards-pass agrees.
    #     @test Zygote.forward(logdet, C)[1] == logdet(C)

    #     # Check that reverse-pass agrees with dense revese-pass.
    #     @testset "logdet gradients" begin
    #         ȳ = randn(rng)

    #         # Compute adjoint with dense.
    #         _, back_dense = Zygote.forward(logdet, C_)
    #         C̄_ = back_dense(ȳ)

    #         # Compute adjoint with block
    #         _, back_block = Zygote.forward(logdet, C)
    #         C̄ = back_block(ȳ)

    #         # Check that both answers approximately agree.
    #         @test C̄_[1].factors ≈ C̄[1].factors
    #     end

    #     @testset "cholesky backsolving" begin
    #         x = BlockVector([randn(rng, P) for P in Ps])
    #         X = BlockMatrix([randn(rng, P, Q) for P in Ps, Q in Qs])
    #         @test Vector(ldiv!(C, copy(x))) ≈ ldiv!(C_, Vector(x))
    #         @test Matrix(ldiv!(C, copy(X))) ≈ ldiv!(C_, Matrix(X))
    #     end
    # end
end
