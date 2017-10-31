@testset "posterior" begin

    import Stheno: LeftFinite, PosteriorData, PosteriorKernel, Constant

    # Test that PosteriorData works as expected.
    let rng = MersenneTwister(123456)
        x = randn(rng, 10, 10)
        U = chol(x'x + 1e-9I)
        data = Stheno.PosteriorData(U)
        @test data.U == U
        @test data.idx == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        @test length(data.tmp) == 10
        @test length(data.tmp′) == 10
    end

    # A couple of posterior checks.
    let rng = MersenneTwister(123456), N = 5

        x = randn(rng, N)
        k1, k2, k12 = EQ(), RQ(1.0), Constant(0.0)
        data, k1f̂ = PosteriorData(chol(k1.(x, x') + 1e-9I)), LeftFinite(k1, x)

        kpost_1 = PosteriorKernel(k1, k1f̂, k1f̂, data)
        @test all(abs.(kpost_1.(x, RowVector(x))) .< 1e-8)

        k2f̂ = LeftFinite(k12, x)
        kpost_21 = PosteriorKernel(k2, k2f̂, k2f̂, data)
        @test all(abs.(kpost_21.(x, RowVector(x)) .- RQ(1.0).(x, RowVector(x))) .< 1e-12)
    end
end
