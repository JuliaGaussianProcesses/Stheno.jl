using Random, LinearAlgebra, BlockArrays
using BlockArrays: _BlockArray

function dense_BlockMatrix_BlockVector_mul_tests(rng, X, y)
    Ps = blocksizes(X, 1)
    z = _BlockArray([randn(rng, P) for P in Ps], Ps)

    @test mul!(z, X, y) isa AbstractBlockVector
    @test Vector(mul!(z, X, y)) ≈ Matrix(X) * Vector(y)

    @test X * y isa AbstractBlockVector
    @test Vector(X * y) ≈ Matrix(X) * Vector(y)

    z̄ = _BlockArray([randn(rng, P) for P in Ps], Ps)
    z, back = Zygote.forward(*, X, y)
    X̄, ȳ = back(z̄)
    @test X̄ isa AbstractBlockMatrix
    @test ȳ isa AbstractBlockVector

    z_dense, back_dense = Zygote.forward(*, Matrix(X), Vector(y))
    X̄_dense, ȳ_dense = back_dense(Vector(z̄))
    @test Vector(z) ≈ z_dense
    @test Matrix(X̄) ≈ X̄_dense
    @test Vector(ȳ) ≈ ȳ_dense

    X̄, ȳ = back(Vector(z̄))
    @test Matrix(X̄) ≈ X̄_dense
    @test Vector(ȳ) ≈ ȳ_dense
end

function dense_BlockMatrix_BlockMatrix_mul_tests(rng, X, Y)
    Ps, Qs = blocksizes(X, 1), blocksizes(Y, 2)
    Z = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)

    @test mul!(Z, X, Y) isa AbstractBlockMatrix
    @test Matrix(mul!(Z, X, Y)) ≈ Matrix(X) * Matrix(Y)

    @test X * Y isa AbstractBlockMatrix
    @test Matrix(X * Y) ≈ Matrix(X) * Matrix(Y)

    Z̄ = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)
    Z, back = Zygote.forward(*, X, Y)
    X̄, Ȳ = back(Z̄)
    @test X̄ isa AbstractBlockMatrix
    @test Ȳ isa AbstractBlockMatrix

    Z_dense, back_dense = Zygote.forward(*, Matrix(X), Matrix(Y))
    X̄_dense, Ȳ_dense = back_dense(Matrix(Z̄))
    @test Matrix(Z) ≈ Z_dense
    @test Matrix(X̄) ≈ X̄_dense
    @test Matrix(Ȳ) ≈ Ȳ_dense

    X̄, Ȳ = back(Matrix(Z̄))
    @test Matrix(X̄) ≈ X̄_dense
    @test Matrix(Ȳ) ≈ Ȳ_dense
end

@testset "fdm stuff" begin
    rng, Ps, Qs = MersenneTwister(123456), [5, 4], [3, 2, 1]
    X = _BlockArray([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)
    vec_X, from_vec = FiniteDifferences.to_vec(X)
    @test vec_X isa Vector
    @test from_vec(vec_X) == X
end
