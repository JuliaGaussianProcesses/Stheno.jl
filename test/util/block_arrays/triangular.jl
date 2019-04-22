using Random, LinearAlgebra, BlockArrays
using BlockArrays: _BlockArray
using Stheno: BlockLowerTriangular, BlockUpperTriangular

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
    @testset "mul! / *" begin
        @test 1 === 0 # The implementation for mul!(y, ::LowerTriangular, x) is clearly bullshit at the minute.
    end
    @testset "ldiv! / \\" begin

    end
end
