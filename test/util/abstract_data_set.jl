using Random, LinearAlgebra, Zygote
using Stheno: ColsAreObs, BlockData

@timedtestset "abstract_data_set" begin
    rng, N, D = MersenneTwister(123456), 10, 2
    x, X = randn(rng, N), randn(rng, D, N)

    # Test Matrix data sets.
    @timedtestset "ColsAreObs" begin
        DX = ColsAreObs(X)
        @test DX == DX
        @test size(DX) == (N,)
        @test length(DX) == N
        @test getindex(DX, 5) isa Vector
        @test getindex(DX, 5) == X[:, 5]
        @test getindex(DX, 1:2:6) isa ColsAreObs
        @test getindex(DX, 1:2:6) == ColsAreObs(X[:, 1:2:6])
        @test view(DX, 4) isa AbstractVector
        @test view(DX, 4) == view(X, :, 4)
        @test view(DX, 1:2:4) isa ColsAreObs
        @test view(DX, 1:2:4) == ColsAreObs(view(X, :, 1:2:4))
        @test eltype(DX) == Vector{Float64}
        @test eachindex(DX) == 1:N

        let
            @test Zygote.forward(ColsAreObs, X)[1] == DX
            DX, back = Zygote.forward(ColsAreObs, X)
            @test back((X=ones(size(X)),))[1] == ones(size(X))

            @test Zygote.forward(DX->DX.X, DX)[1] == X
            X_, back = Zygote.forward(DX->DX.X, DX)
            @test back(ones(size(X)))[1].X == ones(size(X))
        end
    end

    # Test BlockDataSet.
    @timedtestset "BlockData" begin
        DX = ColsAreObs(X)
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
        @test eachindex(DxX) == _BlockArray([1:N, N+1:2N], [N, N])

        # Test iteration.
        @test [x for x in DxX][1] == x[1]
        @test [x for x in DxX][N] == x[end]
        @test [x for x in DxX][N + 1] == DX[1]
        @test [x for x in DxX][end] == DX[end]

        # Test that identically typed arrays yield the appropriate element type.
        @test eltype(BlockData([randn(5), randn(4)])) == Float64
        @test eltype(DxX) == Any
        @test eltype(BlockData([DX, DX])) == Vector{Float64}
        @test eltype(BlockData([eachindex(DX), eachindex(DX)])) == Int
    end
end
