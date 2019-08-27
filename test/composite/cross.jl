using Stheno: BlockData, GPC, cross

@testset "cross" begin
    @testset "Correctness tests" begin
        rng, P, Q, gpc = MersenneTwister(123456), 2, 3, GPC()

        f1 = GP(sin, EQ(), gpc)
        f2 = GP(cos, EQ(), gpc)
        f3 = cross([f1, f2])
        f4 = cross([f1])
        f5 = cross([f2])

        x1 = collect(range(-5.0, 5.0; length=P))
        x2 = collect(range(-5.0, 5.0; length=Q))
        x3 = BlockData([x1, x2])
        x4 = BlockData([x1])
        x5 = BlockData([x2])

        # mean
        @test mean(f3(x3)) == vcat(mean(f1(x1)), mean(f2(x2)))
        @test mean(f4(x4)) == mean(f1(x1))
        @test mean(f5(x5)) == mean(f2(x2))

        # cov
        @test cov(f3(x3)) ≈ vcat(
            hcat(cov(f1(x1)), cov(f1(x1), f2(x2))),
            hcat(cov(f2(x2), f1(x1)), cov(f2(x2))),
        )
        @test cov(f3(x3), f3(x3)) == cov(f3(x3))
        @test cov(f4(x4), f4(x4)) == cov(f4(x4))
        @test cov(f5(x5), f5(x5)) == cov(f5(x5))
        @test cov(f4(x4)) == cov(f1(x1))
        @test cov(f5(x5)) == cov(f2(x2))

        # cross-cov (with block)
        @test cov(f1(x1), f2(x2)) == cov(f4(x4), f5(x5))
        @test cov(f2(x2), f1(x1)) == cov(f5(x5), f4(x4))
        @test cov(f3(x3), f4(x4)) == vcat(cov(f1(x1)), cov(f2(x2), f1(x1)))

        @test cov(f5(x5), f3(x3)) == hcat(cov(f2(x2), f1(x1)), cov(f2(x2)))

        # cross-cov (with non-block)
        @test cov(f4(x4)) == cov(f1(x1))
        @test cov(f5(x5)) == cov(f2(x2))
        @test cov(f3(x3), f4(x4)) == cov(f3(x3), f1(x1))
        @test cov(f5(x5), f3(x3)) == cov(f2(x2), f3(x3))

        @testset "rand, logpdf, elbo" begin

            # Single-sample rand
            y1, y2 = rand([f1(x1), f2(x2)])
            @test length(y1) == length(x1)
            @test length(y2) == length(x2)

            # Multi-sample rand
            Y1, Y2 = rand([f1(x1), f2(x2)], 11)
            @test size(Y1) == (length(y1), 11)
            @test size(Y2) == (length(y2), 11)

            # logpdf
            @test logpdf([f1(x1), f2(x2)], [y1, y2]) ≈ logpdf(f1(x1), y1) + logpdf(f2(x2), y2)
            @test logpdf([f1(x1), f2(x2)], [y1, y2]) ≈ logpdf([f1(x1)←y1, f2(x2)←y2])

            # elbo
            @test elbo([f1(x1, 1e-3), f2(x2, 1e-3)], [y1, y2], [f1(x1), f2(x2)]) ≈
                logpdf([f1(x1, 1e-3), f2(x2, 1e-3)], [y1, y2])
            @test elbo(
                [f1(x1, 1e-3), f2(x2, 1e-3)],
                [y1, y2],
                Stheno.finites_to_block([f1(x1), f2(x2)]),
            ) ≈ logpdf([f1(x1, 1e-3), f2(x2, 1e-3)], [y1, y2])
            @test elbo(f1(x1, 1e-3), y1, [f1(x1), f2(x2)]) ≈ logpdf(f1(x1, 1e-3), y1)
        end
    end
    @testset "Standardised Tests" begin
        rng, P, Q = MersenneTwister(123456), 3, 5
        x0_1, x0_2 = collect(range(-1.0, 1.0; length=P)), collect(range(2.0, 4.0; length=P))
        x1_1, x1_2 = randn(rng, Q), randn(rng, Q)
        x0, x1 = BlockData([x0_1, x0_2]), BlockData([x1_1, x1_2])
        x2, x3 = randn(rng, P), randn(2P)

        gpc = GPC()
        f1, f2 = GP(sin, EQ(), gpc), GP(cos, EQ(), gpc)
        f3 = cross([f1, f2])
        abstractgp_interface_tests(f3, f1, x0, x1, x2, x3)

        # x1, x2 = collect(range(-2.0, 2.0; length=5)), collect(range(1.2, 1.5; length=4))
        # z1, z2 = collect(range(-1.5, 0.75; length=3)), collect(range(0.89, 2.0; length=4))
        # standard_1D_tests(
        #     MersenneTwister(123456),
        #     Dict(:l1=>0.5, :l2=>2.3),
        #     θ->begin
        #         gpc = GPC()
        #         f1 = θ[:l1] * GP(sin, EQ(), gpc)
        #         f2 = θ[:l2] * GP(cos, EQ(), gpc)
        #         f3 = cross([f1, f2])
        #         return f3, f3
        #     end,
        #     BlockData([x1, x2]),
        #     BlockData([z1, z2]),
        # )
    end
    @testset "finites_to_block" begin
        rng, P, Q = MersenneTwister(123456), 3, 5
        xp, xq = randn(rng, P), randn(rng, Q)
        gpc = GPC()
        f_, g_ = GP(sin, EQ(), gpc), GP(cos, EQ(), gpc)
        h_x = Stheno.finites_to_block([f_(xp, 1.0), g_(xq, 1.0)])
    end
end
