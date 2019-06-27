using Stheno: BlockData, GPC, cross

@testset "cross" begin
    # @testset "Correctness tests" begin
    #     rng, P, Q, gpc = MersenneTwister(123456), 11, 13, GPC()

    #     f1 = GP(sin, eq(), gpc)
    #     f2 = GP(cos, eq(), gpc)
    #     f3 = cross([f1, f2])
    #     f4 = cross([f1])
    #     f5 = cross([f2])

    #     x1 = collect(range(-5.0, 5.0; length=P))
    #     x2 = collect(range(-5.0, 5.0; length=Q))
    #     x3 = BlockData([x1, x2])
    #     x4 = BlockData([x1])
    #     x5 = BlockData([x2])

    #     # mean
    #     @test mean(f3(x3)) == vcat(mean(f1(x1)), mean(f2(x2)))
    #     @test mean(f4(x4)) == mean(f1(x1))
    #     @test mean(f5(x5)) == mean(f2(x2))

    #     # cov
    #     @test cov(f3(x3)) == vcat(
    #         hcat(cov(f1(x1)), cov(f1(x1), f2(x2))),
    #         hcat(cov(f2(x2), f1(x1)), cov(f2(x2))),
    #     )
    #     @test cov(f3(x3), f3(x3)) == cov(f3(x3))
    #     @test cov(f4(x4), f4(x4)) == cov(f4(x4))
    #     @test cov(f5(x5), f5(x5)) == cov(f5(x5))
    #     @test cov(f4(x4)) == cov(f1(x1))
    #     @test cov(f5(x5)) == cov(f2(x2))

    #     # cross-cov (with block)
    #     @test cov(f1(x1), f2(x2)) == cov(f4(x4), f5(x5))
    #     @test cov(f2(x2), f1(x1)) == cov(f5(x5), f4(x4))
    #     @test cov(f3(x3), f4(x4)) == vcat(cov(f1(x1)), cov(f2(x2), f1(x1)))
    #     @test cov(f5(x5), f3(x3)) == hcat(cov(f2(x2), f1(x1)), cov(f2(x2)))

    #     # cross-cov (with non-block)
    #     @test cov(f4(x4)) == cov(f1(x1))
    #     @test cov(f5(x5)) == cov(f2(x2))
    #     @test cov(f3(x3), f4(x4)) == cov(f3(x3), f1(x1))
    #     @test cov(f5(x5), f3(x3)) == cov(f2(x2), f3(x3))
    # end
    @testset "Standardised Tests" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:l1=>0.5, :l2=>2.3),
            θ->begin
                gpc = GPC()
                f1 = GP(sin, eq(l=θ[:l1]), gpc)
                f2 = GP(cos, eq(l=θ[:l2]), gpc)
                f3 = cross([f1, f2])
                return f3, f3
            end,
            13, 11,
        )
    end
    # @testset "finites_to_block" begin
    #     rng, P, Q = MersenneTwister(123456), 11, 13
    #     xp, xq = randn(rng, P), randn(rng, Q)
    #     gpc = GPC()
    #     f_, g_ = GP(sin, eq(), gpc), GP(cos, eq(), gpc)
    #     h_x = Stheno.finites_to_block([f_(xp, 1.0), g_(xq, 1.0)])
    #     @show size(h_x.x), size(h_x.Σy)
    # end
end
