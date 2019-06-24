using Stheno: BlockData, GPC, cross

@testset "cross" begin
    @testset "Correctness tests" begin
        rng, P, Q, gpc = MersenneTwister(123456), 11, 13, GPC()

        f1 = GP(sin, eq(), gpc)
        f2 = GP(cos, eq(), gpc)
        f3 = cross([f1, f2])

        x1 = collect(range(-5.0, 5.0; length=P))
        x2 = collect(range(-5.0, 5.0; length=Q))
        x3 = BlockData([x1, x2])

        @test mean(f3(x3)) == vcat(mean(f1(x1)), mean(f2(x2)))
    end
end
