using Stheno: AbstractDataSet, DataSet, BlockDataSet, VectorData, MatrixData

@testset "util/abstract_data_set" begin

    let
        rng, N, D = MersenneTwister(123456), 10, 2
        x, X = randn(rng, N), randn(rng, D, N)

        @test IndexStyle(AbstractDataSet) == IndexLinear()

        # Test Vector data sets.
        Dx = DataSet(x)
        @test size(Dx) == (N,)
        @test length(Dx) == N
        @test Dx == Dx
        @test getindex(Dx, 5) isa Real
        @test getindex(Dx, 5) == x[5]
        @test getindex(Dx, 1:2:6) isa VectorData
        @test getindex(Dx, 1:2:6) == DataSet(x[1:2:6])
        @test view(Dx, 4) isa AbstractArray
        @test view(Dx, 4) == view(x, 4)
        @test view(Dx, 1:2:6) isa VectorData
        @test view(Dx, 1:2:6) == DataSet(view(x, 1:2:6))

        @test eltype(Dx) == Float64
        @test eachindex(Dx) == 1:N

        # Iteration.
        @test start(Dx) == 1
        @test next(Dx, 1) == (x[1], 2)
        @test endof(Dx) == length(Dx)
        @test [x_ for x_ in Dx] == x

        # Test Matrix data sets.
        DX = DataSet(X)
        @test DX == DX
        @test size(DX) == (N,)
        @test length(DX) == N
        @test getindex(DX, 5) isa Vector
        @test getindex(DX, 5) == X[:, 5]
        @test getindex(DX, 1:2:6) isa MatrixData
        @test getindex(DX, 1:2:6) == DataSet(X[:, 1:2:6])
        @test view(DX, 4) isa AbstractVector
        @test view(DX, 4) == view(X, :, 4)
        @test view(DX, 1:2:4) isa MatrixData
        @test view(DX, 1:2:4) == DataSet(view(X, :, 1:2:4))
        @test eltype(DX) == Vector{Float64}
        @test eachindex(DX) == 1:N

        # Iteration.
        @test start(DX) == 1
        @test endof(DX) == length(DX)
        @test next(DX, 1) == (DX[1], 2)
        @test hcat([x for x in DX]...) == X

        # Test BlockDataSets.
        DxX = BlockDataSet([x, X])
        @test size(DxX) == (2N,)
        @test length(DxX) == 2N
        @test DxX == DxX
        @test DxX == BlockDataSet([Dx, DX])
        @test endof(DxX) == (2, N)
        @test length([x for x in DxX]) == 2N
        @test getindex(DxX, 1, 1) == Dx[1]
        @test getindex(DxX, 1, N) == Dx[end]
        @test getindex(DxX, 2, 1) == DX[1]
        @test getindex(DxX, 2, N) == DX[end]
        @test getindex(DxX, 1) === Dx
        @test getindex(DxX, 2) === DX
        @test view(DxX, 2, 1) == view(DX, 1)
        @test view(DxX, 2, N) == view(DX, N)
        @test eachindex(DxX) == 1:2N

        # Test iteration.
        @test start(DxX) == (1, 1)
        @test next(DxX, (1, 1)) == (x[1], (1, 2))
        @test next(DxX, (1, N)) == (x[end], (1, N + 1))
        @test next(DxX, (1, N + 1)) == (DX[1], (2, 2))
        @test [x for x in DxX][1] == Dx[1]
        @test [x for x in DxX][N] == Dx[end]
        @test [x for x in DxX][N + 1] == DX[1]
        @test [x for x in DxX][end] == DX[end]
    end
end
