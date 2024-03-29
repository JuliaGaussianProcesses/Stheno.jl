@timedtestset "cross" begin
    @timedtestset "block arrays" begin
        @timedtestset "fdm stuff" begin
            rng, Ps, Qs = MersenneTwister(123456), [5, 4], [3, 2, 1]
            X = mortar([randn(rng, P, Q) for P in Ps, Q in Qs], Ps, Qs)
            vec_X, from_vec = FiniteDifferences.to_vec(X)
            @test vec_X isa Vector
            @test from_vec(vec_X) == X
        end
        @timedtestset "Stheno._collect ∘ _mortar" begin
            @timedtestset "BlockVector" begin

                # Generate some blocks.
                Ps = [5, 6, 7]
                x = BlockArray(randn(sum(Ps)), Ps).blocks

                # Verify the pullback.
                ȳ = randn(sum(Ps))
                adjoint_test(Stheno._collect ∘ Stheno._mortar, ȳ, x)
            end
            @timedtestset "BlockMatrix" begin

                # Generate some blocks.
                Ps = [3, 4, 5]
                Qs = [6, 7, 8, 9]
                X = BlockArray(randn(sum(Ps), sum(Qs)), Ps, Qs).blocks
                Ȳ = randn(sum(Ps), sum(Qs))

                # Verify pullback.
                adjoint_test(Stheno._collect ∘ Stheno._mortar, Ȳ, X)
            end
        end
    end
    @timedtestset "Correctness tests" begin
        rng, P, Q, gpc = MersenneTwister(123456), 2, 3, GPC()

        f1 = atomic(GP(sin, SEKernel()), gpc)
        f2 = atomic(GP(cos, SEKernel()), gpc)
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
    end
    @timedtestset "Standardised Tests" begin
        rng, P, Q = MersenneTwister(123456), 3, 5
        x0_1, x0_2 = collect(range(-1.0, 1.0; length=P)), collect(range(2.0, 4.0; length=P))
        x1_1, x1_2 = randn(rng, Q), randn(rng, Q)
        x0, x1 = BlockData([x0_1, x0_2]), BlockData([x1_1, x1_2])
        x2, x3 = randn(rng, P), randn(2P)

        gpc = GPC()
        f1 = atomic(GP(sin, SEKernel()), gpc)
        f2 = atomic(GP(cos, SEKernel()), gpc)
        f3 = cross([f1, f2])
        abstractgp_interface_tests(f3, f1, x0, x1, x2, x3)

        # x1, x2 = collect(range(-2.0, 2.0; length=5)), collect(range(1.2, 1.5; length=4))
        # z1, z2 = collect(range(-1.5, 0.75; length=3)), collect(range(0.89, 2.0; length=4))
        # standard_1D_tests(
        #     MersenneTwister(123456),
        #     Dict(:l1=>0.5, :l2=>2.3),
        #     θ->begin
        #         gpc = GPC()
        #         f1 = θ[:l1] * GP(sin, SqExponentialKernel(), gpc)
        #         f2 = θ[:l2] * GP(cos, SqExponentialKernel(), gpc)
        #         f3 = cross([f1, f2])
        #         return f3, f3
        #     end,
        #     BlockData([x1, x2]),
        #     BlockData([z1, z2]),
        # )
    end
end
