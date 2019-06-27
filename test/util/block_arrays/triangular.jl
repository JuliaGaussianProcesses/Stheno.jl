using Random, LinearAlgebra, BlockArrays
using BlockArrays: _BlockArray
using Stheno: BlockLowerTriangular, BlockUpperTriangular
using LinearAlgebra: ldiv!, \

function solve_BlockTriangular_Vector_tests(rng, T, x, TriangleType)
    Ps, Tmat, xvec = blocksizes(T, 1), TriangleType(Matrix(T)), Vector(x)

    @test ldiv!(T, copy(x)) isa BlockVector
    @test Vector(ldiv!(T, copy(x))) ≈ ldiv!(Tmat, copy(xvec))
    @test T \ x isa BlockVector
    @test ldiv!(T, copy(x)) == T \ x

    z̄ = _BlockArray([randn(rng, P) for P in Ps], Ps)

    z, back = Zygote.forward(\, T, x)
    L̄, x̄ = back(z̄)
    @test L̄ isa BlockMatrix
    @test x̄ isa BlockVector

    z_vec, back_mat = Zygote.forward(\, Tmat, xvec)
    L̄_mat, x̄_vec = back_mat(Vector(z̄))
    @test z_vec ≈ Vector(z)
    @test L̄_mat ≈ Matrix(L̄)
    @test x̄_vec ≈ Vector(x̄)
end

function solve_BlockTriangular_Matrix_tests(rng, T, X, TriangleType)
    Ps, Qs = blocksizes(X, 1), blocksizes(X, 2)
    Tmat, Xmat = TriangleType(Matrix(T)), Matrix(X)

    @test ldiv!(T, copy(X)) isa BlockMatrix
    @test Matrix(ldiv!(T, copy(X))) ≈ ldiv!(Tmat, copy(Xmat))
    @test T \ X isa BlockMatrix
    @test ldiv!(T, copy(X)) == T \ X

    Z̄ = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

    Z, back = Zygote.forward(\, T, X)
    T̄, X̄ = back(Z̄)
    @test T̄ isa BlockMatrix
    @test X̄ isa BlockMatrix

    Z_vec, back_mat = Zygote.forward(\, Tmat, Xmat)
    T̄_mat, X̄_mat = back_mat(Matrix(Z̄))
    @test Z_vec ≈ Matrix(Z)
    @test T̄_mat ≈ Matrix(T̄)
    @test X̄_mat ≈ Matrix(X̄)
end

@testset "triangular" begin
    @testset "adjoint / transpose" begin
        rng, Ps = MersenneTwister(123456), [5, 4]
        X = _BlockArray([randn(rng, P, P′) for P in Ps, P′ in Ps], Ps, Ps)
        Xmat = Matrix(X)
        for foo in [adjoint, transpose]
            @test foo(UpperTriangular(X)) isa BlockLowerTriangular
            @test Matrix(foo(UpperTriangular(X))) == collect(foo(UpperTriangular(Xmat)))
            @test foo(LowerTriangular(X)) isa BlockUpperTriangular
            @test Matrix(foo(LowerTriangular(X))) == collect(foo(LowerTriangular(Xmat)))
        end
    end
    @testset "construction" begin
        rng, Ps = MersenneTwister(123456), [5, 6, 7]
        @testset "BlockLowerTriangular" begin
            L = LowerTriangular(randn(rng, sum(Ps), sum(Ps)))
            @test BlockArray(L, Ps, Ps) isa BlockLowerTriangular
            @test BlockArray(L, Ps, Ps) == L
        end
        @testset "BlockUpperTriangular" begin
            U = UpperTriangular(randn(rng, sum(Ps), sum(Ps)))
            @test BlockArray(U, Ps, Ps) isa BlockUpperTriangular
            @test BlockArray(U, Ps, Ps) == U
        end 
    end
    @testset "copy" begin
        rng, Ps = MersenneTwister(123456), [5, 4]
        X = BlockArray(randn(rng, sum(Ps), sum(Ps)), Ps, Ps)

        @testset "BlockLowerTriangular" begin 
            L = LowerTriangular(X)
            @test copy(L) isa BlockLowerTriangular
            @test copy(L) == L
        end
        @testset "BlockUpperTriangular" begin
            U = UpperTriangular(X)
            @test copy(U) isa BlockUpperTriangular
            @test copy(U) == U
        end
    end
    @testset "mul! / *" begin
        rng, Ps, Qs = MersenneTwister(123456), [5, 4, 3], [7, 6, 5]
        A = _BlockArray([randn(rng, P, P′) for P in Ps, P′ in Ps], Ps, Ps)
        L, U = LowerTriangular(A), UpperTriangular(A)
        x = _BlockArray([randn(rng, P) for P in Ps], Ps)
        X = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

        @testset "LowerTriangular" begin
            @testset "Matrix-Vector" begin
                dense_BlockMatrix_BlockVector_mul_tests(rng, L, x)
            end
            @testset "Matrix-Matrix" begin
                dense_BlockMatrix_BlockMatrix_mul_tests(rng, L, X)
            end
        end
        @testset "UpperTriangular" begin
            @testset "Matrix-Vector" begin
                dense_BlockMatrix_BlockVector_mul_tests(rng, U, x)
            end
            @testset "Matrix-Matrix" begin
                dense_BlockMatrix_BlockMatrix_mul_tests(rng, U, X)
            end
        end
    end
    @testset "ldiv! / \\" begin
        rng, Ps, Qs = MersenneTwister(123456), [5, 4, 3], [7, 6, 5]
        A = _BlockArray([randn(rng, P, P′) for P in Ps, P′ in Ps], Ps, Ps)
        L, U = LowerTriangular(A), UpperTriangular(A)
        x = _BlockArray([randn(rng, P) for P in Ps], Ps)
        X = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

        @testset "LowerTriangular" begin
            @testset "Matrix-Vector" begin
                solve_BlockTriangular_Vector_tests(rng, L, x, LowerTriangular)
            end
            @testset "Matrix-Matrix" begin
                solve_BlockTriangular_Matrix_tests(rng, L, X, LowerTriangular)
            end
        end
        @testset "UpperTriangular" begin
            @testset "Matrix-Vector" begin
                solve_BlockTriangular_Vector_tests(rng, U, x, UpperTriangular)
            end
            @testset "Matrix-Matrix" begin
                solve_BlockTriangular_Matrix_tests(rng, U, X, UpperTriangular)
            end
        end
    end
end
