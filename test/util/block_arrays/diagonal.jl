function BlockDiagonal_mul_tests(rng, blocks)
    D, Ps = block_diagonal(blocks), size.(blocks, 1)
    Dmat = Matrix(D)

    U = UpperTriangular(D)

    xs, ys = [randn(rng, P) for P in Ps], [randn(rng, P) for P in Ps]
    y, x = _BlockArray(ys, Ps), _BlockArray(xs, Ps)

    # Matrix-Vector product
    @test mul!(y, D, x) ≈ Dmat * Vector(x)
    @test mul!(y, D, x) == D * x
    @test mul!(y, U, x) ≈ Matrix(U) * Vector(x)
    @test mul!(y, U, x) == U * x

    # ȳ = D * x
    # adjoint_test(*, ȳ, D, x)
    # let
    #     y, back = Zygote.forward(*, D, x)
    #     @test back(ȳ) isa BlockVector
    # end
    # adjoint_test(*, ȳ, U, x)
    # let
    #     y, back = Zygote.forward(*, U, x)
    #     @test back(ȳ) isa BlockVector
    # end

    Qs = [3, 4]
    X = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)
    Y = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

    # Matrix-Matrix product
    @test mul!(Y, D, X) ≈ Dmat * X
    @test mul!(Y, U, X) ≈ Matrix(U) * Matrix(X)
    @test mul!(Y, D, X) == D * X
    @test mul!(Y, U, X) == U * X
end

function BlockDiagonal_solve_tests(rng, blocks)

    D, Ps = block_diagonal(blocks), size.(blocks, 1)
    Dmat = Matrix(D)

    U = UpperTriangular(D)

    xs, ys = [randn(rng, P) for P in Ps], [randn(rng, P) for P in Ps]
    y, x = _BlockArray(ys, Ps), _BlockArray(xs, Ps)

    # Matrix-Vector tests.
    @test ldiv!(U, copy(x)) ≈ UpperTriangular(Matrix(U)) \ Vector(x)
    @test U \ x ≈ ldiv!(U, copy(x))

    Qs = [3, 4]
    X = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)
    Y = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

    # Matrix-Matrix tests.
    @test ldiv!(U, copy(X)) ≈ UpperTriangular(Matrix(U)) \ Matrix(X)
    @test U \ X ≈ ldiv!(U, copy(X))
end

function BlockDiagonal_chol_tests(rng, blocks)

    D, Ps = block_diagonal(blocks), size.(blocks, 1)
    Dmat = Matrix(D)

    C, Cmat = cholesky(D), cholesky(Dmat)

    @test C.U ≈ Cmat.U
    @test logdet(C) ≈ logdet(Cmat)

    Csym = cholesky(Symmetric(D))
    @test C.U ≈ Csym.U
end

@testset "BlockDiagonal" begin
    @testset "Matrix" begin
        rng, Ps, Qs = MersenneTwister(123456), [2, 3], [4, 5]
        vs = [randn(rng, Ps[1], Qs[1]), randn(rng, Ps[2], Qs[2])]
        general_BlockDiagonal_tests(rng, vs)

        As = [randn(rng, Ps[n], Ps[n]) for n in eachindex(Ps)]
        blocks = [As[n] * As[n]' + I for n in eachindex(As)]
        BlockDiagonal_mul_tests(rng, blocks)
        BlockDiagonal_mul_tests(rng, UpperTriangular.(blocks))
        BlockDiagonal_mul_tests(rng, Hermitian.(blocks))
        BlockDiagonal_mul_tests(rng, Symmetric.(blocks))
        BlockDiagonal_solve_tests(rng, UpperTriangular.(blocks))
        BlockDiagonal_chol_tests(rng, blocks)
    end
    @testset "Diagonal{T, <:Vector{T}}" begin
        rng, Ps = MersenneTwister(123456), [2, 3]
        vs = [Diagonal(randn(rng, Ps[n])) for n in eachindex(Ps)]
        general_BlockDiagonal_tests(rng, vs)

        blocks = [Diagonal(ones(P) + exp.(randn(rng, P))) for P in Ps]
        BlockDiagonal_mul_tests(rng, blocks)
        BlockDiagonal_solve_tests(rng, blocks)
        BlockDiagonal_chol_tests(rng, blocks)
    end
    @testset "Diagonal{T, <:Fill{T, 1}}" begin
        rng, Ps = MersenneTwister(123456), [2, 3]
        vs = [Diagonal(Fill(randn(rng), P)) for P in Ps]
        general_BlockDiagonal_tests(rng, vs)

        blocks = [Diagonal(Fill(exp(randn(rng)), P)) for P in Ps]
        BlockDiagonal_mul_tests(rng, blocks)
        BlockDiagonal_solve_tests(rng, blocks)
        BlockDiagonal_chol_tests(rng, blocks)
    end
end
