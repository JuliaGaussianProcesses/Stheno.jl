@testset "lin_ops" begin

    # Test indexing into a GP and that the cross-covariances with another independent GP
    # are zero.
    let rng = MersenneTwister(123456)

        # Set up some GPs.
        x, y = randn(rng, 3), randn(rng, 2)
        μ1, μ2 = sin, cos
        k1, k2 = EQ(), RQ(10.0)
        gpc = GPC()
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
        f3 = f1(x)

        # Check mean and marginal covariance under indexing.        
        @test mean(f3).(eachindex(x)) == mean(f1).(x)
        for m in eachindex(x), n in eachindex(x)
            @test kernel(f3)(m, n) == kernel(f1)(x[m], x[n])
        end
        for m in eachindex(x), n in eachindex(y)
            @test kernel(f3, f1)(m, y[n]) == kernel(f1)(x[m], y[n])
            @test kernel(f1, f3)(y[n], m) == kernel(f1)(y[n], x[m])

            @test kernel(f3, f2)(m, y[n]) == 0.0
            @test kernel(f2, f3)(y[n], m) == 0.0
        end

        @test memory(@benchmark $(mean(f3))(1) seconds=0.1) == 0
        @test memory(@benchmark $(kernel(f3))(1, 2) seconds=0.1) == 0
    end

    # Test inference.
    let rng = MersenneTwister(123456)
        x, x′, f̂ = randn(rng, 3), randn(rng, 2), randn(rng, 3)
        f = GP(sin, EQ(), GPC())
        fpost_d = posterior(f(x), f(x), f̂)
        fpost_d′ = posterior(f(x′), f(x), f̂)
        fpost_gp = posterior(f, f(x), f̂)

        # Test finite GP posterior.
        idx = eachindex(fpost_d)
        @test dims(fpost_d) == length(x)
        @test mean(fpost_d).(idx) ≈ f̂
        @test all(full(cov(fpost_d)) .- diagm(2e-9 * ones(x)) .< 1e-12)
        @test dims(fpost_d′) == length(x′)

        # Test process posterior works.
        @test mean(fpost_gp).(x) ≈ f̂
        @test all(full(cov(fpost_gp(x))) .- diagm(2e-9 * ones(x)) .< 1e-12)
    end

    # Test addition of GPs.
    let rng = MersenneTwister(123456)

        # Select some input locations.
        x, y = randn(rng, 3), randn(rng, 2)

        # Set three independent GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        gpc = GPC()
        f1, f2, f3 = GP(μ1, k1, gpc), GP(μ2, k2, gpc), GP(μ3, k3, gpc)

        # Compute all four summations between first two GPs.
        f_1p1 = f1 + f1
        f_1p2 = f1 + f2
        f_2p1 = f2 + f1
        f_2p2 = f2 + f2

        # Check that the mean functions have been correctly computed.
        @test mean(f_1p1).(x) == μ1.(x) .+ μ1.(x)
        @test mean(f_1p2).(x) == μ1.(x) .+ μ2.(x)
        @test mean(f_2p1).(x) == μ2.(x) .+ μ1.(x)
        @test mean(f_2p2).(x) == μ2.(x) .+ μ2.(x)

        # Check that the marginal covariances have been correctly computed.
        @test full(cov(kernel(f_1p1), x)) ≈ 4 .* full(cov(k1, x))
        @test full(cov(kernel(f_1p2), x)) ≈ full(cov(k1, x)) .+ full(cov(k2, x))
        @test full(cov(kernel(f_2p1), x)) ≈ full(cov(k2, x)) .+ full(cov(k1, x))
        @test full(cov(kernel(f_2p2), x)) ≈ 4 .* full(cov(k2, x))

        # Check that the cross-covariances have been correctly computed.
        @test full(cov(kernel(f1, f_1p1), x)) ≈ 2 .* full(cov(kernel(f1), x))
        @test full(cov(kernel(f1, f_1p2), x)) ≈ full(cov(kernel(f1), x))
        @test full(cov(kernel(f1, f_2p1), x)) ≈ full(cov(kernel(f1), x))
        @test full(cov(kernel(f1, f_2p2), x)) ≈ diagm(1e-9 * ones(3))

        # Check that the cross-covariances match.
        @test full(cov(kernel(f1, f_1p1), x)) == permutedims(full(cov(kernel(f_1p1, f1), x)), [2, 1])
        @test full(cov(kernel(f1, f_1p2), x)) == permutedims(full(cov(kernel(f_1p2, f1), x)), [2, 1])
        @test full(cov(kernel(f1, f_2p1), x)) == permutedims(full(cov(kernel(f_2p1, f1), x)), [2, 1])
        @test full(cov(kernel(f1, f_2p2), x)) == permutedims(full(cov(kernel(f_2p2, f1), x)), [2, 1])

        # Memory performance tests.
        @test memory(@benchmark $(mean(f_1p1))(1.0) seconds=0.1) == 0
        @test memory(@benchmark $(mean(f_1p2))(0.0) seconds=0.1) == 0
        @test memory(@benchmark $(mean(f_2p1))(-1.0) seconds=0.1) == 0
        @test memory(@benchmark $(mean(f_2p2))(5.0) seconds=0.1) == 0
        @test memory(@benchmark $(kernel(f1, f_1p1))(1.0, 0.0) seconds=0.1) == 0
        @test memory(@benchmark $(kernel(f_2p2))(0.0, 1.0) seconds=0.1) == 0
    end
end
