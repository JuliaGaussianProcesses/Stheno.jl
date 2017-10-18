@testset "lin_ops" begin

    # Test Stheno.get_check_gpc
    let
        # Set up some GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        gpc1, gpc2 = GPCollection(), GPCollection()
        gp1, gp2 = GP.(gpc1, [μ1, μ2], [k1, k2])
        gp3 = GP(gpc2, μ3, k3)

        # Check that they pass / fail as appropriate.
        @test gpc1 === Stheno.get_check_gpc(gp1, gp2)
        @test gpc2 === Stheno.get_check_gpc(gp3)
        @test_throws AssertionError Stheno.get_check_gpc(gp1, gp3)
        @test_throws AssertionError Stheno.get_check_gpc(gp2, gp3)
    end

    # Test indexing into GPs.
    let rng = MersenneTwister(123456)

        # Set up some GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        gp1, gp2, gp3 = GP.(GPCollection(), [μ1, μ2, μ3], [k1, k2, k3])

        # Index into `gp1` and check that stuff lines up.
        N = 2
        x = randn(N)
        @test mean(gp1(x)).(eachindex(x)) == mean(gp1).(x)
        @test mean(gp2(x)).(eachindex(x)) == mean(gp2).(x)
        @test mean(gp3(x)).(eachindex(x)) == mean(gp3).(x)

        for m in 1:N, n in 1:N

            # Check the marginal covariance matrices.
            @test kernel(gp1(x))(m, n) == kernel(gp1)(x[m], x[n])
            @test kernel(gp2(x))(m, n) == kernel(gp2)(x[m], x[n])
            @test kernel(gp3(x))(m, n) == kernel(gp3)(x[m], x[n])

            # Check the cross-covariances one way.
            @test kernel(gp1, gp1(x))(x[m], n) == kernel(gp1)(x[m], x[n])

            # Check the cross-covariances the other way.

        end
    end

    # Test addition of GPs.
    let rng = MersenneTwister(123456)

        # Select some input locations.
        N = 2
        x = randn(rng, N)

        # Set up a pair of GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        gpc = GPCollection()
        gp1 = GP(gpc, μ1, k1)
        gp2 = GP(gpc, μ2, k2)
        gp3 = GP(gpc, μ3, k3)

        # Compute all four summations (we will check for approximate commutativity).
        gp_1p1 = gp1 + gp1
        gp_1p2 = gp1 + gp2
        gp_2p1 = gp2 + gp1
        gp_2p2 = gp2 + gp2

        # Check that the mean functions have been correctly computed.
        @test mean(gp_1p1).(x) == μ1.(x) .+ μ1.(x)
        @test mean(gp_1p2).(x) == μ1.(x) .+ μ2.(x)
        @test mean(gp_2p1).(x) == μ2.(x) .+ μ1.(x)
        @test mean(gp_2p2).(x) == μ2.(x) .+ μ2.(x)

        # Check that the marginal covariances have been correctly computed.
        @test full(cov(kernel(gp_1p1), x)) ≈ 4 .* full(cov(k1, x))
        @test full(cov(kernel(gp_1p2), x)) ≈ full(cov(k1, x)) .+ full(cov(k2, x))
        @test full(cov(kernel(gp_2p1), x)) ≈ full(cov(k2, x)) .+ full(cov(k1, x))
        @test full(cov(kernel(gp_2p2), x)) ≈ 4 .* full(cov(k2, x))

        # Check that the cross-covariances have been correctly computed.
        @test full(cov(kernel(gp1, gp_1p1), x)) ≈ 2 .* full(cov(kernel(gp1), x))
        @test full(cov(kernel(gp1, gp_1p2), x)) ≈ full(cov(kernel(gp1), x))
        @test full(cov(kernel(gp1, gp_2p1), x)) ≈ full(cov(kernel(gp1), x))
        @test full(cov(kernel(gp1, gp_2p2), x)) ≈ diagm(1e-12 * ones(N))

        # Check that the cross-covariances match.
        @test full(cov(kernel(gp1, gp_1p1), x)) == permutedims(full(cov(kernel(gp_1p1, gp1), x)), [2, 1])
        @test full(cov(kernel(gp1, gp_1p2), x)) == permutedims(full(cov(kernel(gp_1p2, gp1), x)), [2, 1])
        @test full(cov(kernel(gp1, gp_2p1), x)) == permutedims(full(cov(kernel(gp_2p1, gp1), x)), [2, 1])
        @test full(cov(kernel(gp1, gp_2p2), x)) == permutedims(full(cov(kernel(gp_2p2, gp1), x)), [2, 1])
    end

end
