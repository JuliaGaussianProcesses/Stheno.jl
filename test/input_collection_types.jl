@testset "input_collection_types" begin
    @timedtestset "BlockData" begin

        rng = MersenneTwister(123456)
        N = 10
        D = 2
        x = randn(rng, N)
        X = randn(rng, D, N)

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

        # Convenience constructors.
        @test BlockData(x, DX) == DxX

        # Convenience constructors when we have GPPPInputs.
        @timedtestset "vcat(::GPPPInput...)" begin
            ax = GPPPInput(:a, x)
            bx = GPPPInput(:b, DX)
            @test vcat(ax, bx) isa BlockData
            @test vcat(ax, bx) == vcat(collect(ax), collect(bx))

            @test eltype(vcat(ax, ax)) == eltype(ax)
        end
    end
end
