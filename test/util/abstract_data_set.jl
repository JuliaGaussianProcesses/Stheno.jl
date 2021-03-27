using Stheno: ColVecs, BlockData

@timedtestset "abstract_data_set" begin
    rng, N, D = MersenneTwister(123456), 10, 2
    x, X = randn(rng, N), randn(rng, D, N)

    # Test BlockDataSet.
    @timedtestset "BlockData" begin
        DX = ColVecs(X)
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
        @test eachindex(DxX) isa BlockVector
        @test eachindex(DxX) == mortar([1:N, N+1:2N], [N, N])

        # Test iteration.
        @test [x for x in DxX][1] == x[1]
        @test [x for x in DxX][N] == x[end]
        @test [x for x in DxX][N + 1] == DX[1]
        @test [x for x in DxX][end] == DX[end]

        # Test that identically typed arrays yield the appropriate element type.
        @test eltype(BlockData([randn(5), randn(4)])) == Float64
        @test eltype(DxX) == Any
        @test eltype(BlockData([DX, DX])) <: AbstractVector{Float64}
        @test eltype(BlockData([eachindex(DX), eachindex(DX)])) == Int
    end
end
