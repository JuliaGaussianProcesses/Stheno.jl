@testset "lin_ops" begin

    # Test indexing into a GP and that the cross-covariances with another independent GP
    # are zero.
    let rng = MersenneTwister(123456)

        # Set up some GPs.
        x, y = randn(rng, 3), randn(rng, 2)
        μ1, μ2 = sin, cos
        k1, k2 = EQ(), RQ(10.0)
        f1, f2 = GP(μ1, k1), GP(μ2, k2)
        f3 = f1(x)

        # Check mean and marginal covariance under indexing.        
        @test mean(f3).(eachindex(x)) == mean(f1).(x)
        for m in eachindex(x), n in eachindex(x)
            @test kernel(f3)(m, n) == kernel(f1)(x[m], x[n])
        end
        for m in eachindex(x), n in eachindex(y)
            @test kernel(f3, f1)(m, y[n]) == kernel(f1)(x[m], y[n])
            @test kernel(f1, f3)(y[n], m) == kernel(f1)(y[n], x[m])
        end
        @test kernel(f3, f2) == Constant(0.0)
        @test kernel(f2, f3) == Constant(0.0)
    end

    # Test addition of GPs.
    let rng = MersenneTwister(123456)

        # Select some input locations.
        x, y = randn(rng, 3), randn(rng, 2)

        # Set three independent GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        f1, f2, f3 = GP(μ1, k1), GP(μ2, k2), GP(μ3, k3)

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

        # # Check that the marginal covariances have been correctly computed.
        # @test full(cov(kernel(gp_1p1), x)) ≈ 4 .* full(cov(k1, x))
        # @test full(cov(kernel(gp_1p2), x)) ≈ full(cov(k1, x)) .+ full(cov(k2, x))
        # @test full(cov(kernel(gp_2p1), x)) ≈ full(cov(k2, x)) .+ full(cov(k1, x))
        # @test full(cov(kernel(gp_2p2), x)) ≈ 4 .* full(cov(k2, x))

        # # Check that the cross-covariances have been correctly computed.
        # @test full(cov(kernel(gp1, gp_1p1), x)) ≈ 2 .* full(cov(kernel(gp1), x))
        # @test full(cov(kernel(gp1, gp_1p2), x)) ≈ full(cov(kernel(gp1), x))
        # @test full(cov(kernel(gp1, gp_2p1), x)) ≈ full(cov(kernel(gp1), x))
        # @test full(cov(kernel(gp1, gp_2p2), x)) ≈ diagm(1e-12 * ones(N))

        # # Check that the cross-covariances match.
        # @test full(cov(kernel(gp1, gp_1p1), x)) == permutedims(full(cov(kernel(gp_1p1, gp1), x)), [2, 1])
        # @test full(cov(kernel(gp1, gp_1p2), x)) == permutedims(full(cov(kernel(gp_1p2, gp1), x)), [2, 1])
        # @test full(cov(kernel(gp1, gp_2p1), x)) == permutedims(full(cov(kernel(gp_2p1, gp1), x)), [2, 1])
        # @test full(cov(kernel(gp1, gp_2p2), x)) == permutedims(full(cov(kernel(gp_2p2, gp1), x)), [2, 1])
    end
end
