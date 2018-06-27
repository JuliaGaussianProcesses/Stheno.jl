using Stheno: AbstractDataSet, MatDataSet, BlockData

@testset "util/abstract_data_set" begin

    let
        rng, N, D = MersenneTwister(123456), 10, 2
        x, X = randn(rng, N), randn(rng, D, N)

        @test IndexStyle(AbstractDataSet) == IndexLinear()

        # Test Matrix data sets.
        DX = MatDataSet(X)
        @test DX == DX
        @test size(DX) == (N,)
        @test length(DX) == N
        @test getindex(DX, 5) isa Vector
        @test getindex(DX, 5) == X[:, 5]
        @test getindex(DX, 1:2:6) isa MatDataSet
        @test getindex(DX, 1:2:6) == MatDataSet(X[:, 1:2:6])
        @test view(DX, 4) isa AbstractVector
        @test view(DX, 4) == view(X, :, 4)
        @test view(DX, 1:2:4) isa MatDataSet
        @test view(DX, 1:2:4) == MatDataSet(view(X, :, 1:2:4))
        @test eltype(DX) == Vector{Float64}
        @test eachindex(DX) == 1:N

        # Iteration.
        @test start(DX) == 1
        @test endof(DX) == length(DX)
        @test next(DX, 1) == (DX[1], 2)
        @test hcat([x for x in DX]...) == X

        # Test BlockDataSet.
        DX = MatDataSet(X)
        @show typeof([x, DX]), eltype([x, DX])
        DxX = BlockData([x, DX])
        @test size(DxX) == (2N,)
        @test length(DxX) == 2N
        @test DxX == DxX
        @test DxX == BlockData([x, DX])
        @test length([x for x in DxX]) == 2N
        @test getindex(DxX, 1) === x[1]
        @test getindex(DxX, 2) === x[2]
        @test getindex(DxX, N) === x[N]
        @test getindex(DxX, N + 1) == DX[1]
        @test view(DxX, 2, 1) == view(DX, 1)
        @test view(DxX, 2, N) == view(DX, N)
        @test eachindex(DxX) == 1:2N
        @show eltype(DxX)

        # Test iteration.
        @test [x for x in DxX][1] == x[1]
        @test [x for x in DxX][N] == x[end]
        @test [x for x in DxX][N + 1] == DX[1]
        @test [x for x in DxX][end] == DX[end]
    end
end
