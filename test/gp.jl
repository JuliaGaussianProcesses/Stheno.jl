@testset "gp" begin

    # Test basics for `KernelCollection` container.
    import Stheno.KernelCollection
    let rng = MersenneTwister(123456)

        r1, r2, r3 = randn(rng, 5), randn(rng, 6), randn(rng, 7)
        jk = KernelCollection([r1, r2, r3])

        @test jk[1, 1] === r1[1]
        @test jk[1, 2] === r2[1]
        @test jk[2, 1] === jk[1, 2]
        @test jk[3, 3] === r3[3]
        @test_throws BoundsError jk[1, 4]
        @test_throws BoundsError jk[4, 1]
        @test size(jk) == (3, 3)
        @test size(jk, 1) == 3

        jk_ext = push!(jk, r1)
        @test jk_ext[1, 4] === r1[1]
        @test jk_ext[4, 1] === jk_ext[1, 4]
        @test size(jk_ext) == (4, 4)
        @test size(jk_ext, 2) == 4
    end

    # Test `length`, `mean`, `kernel` and `push!` for `GPCollection`.
    import Stheno.GPCollection
    let rng = MersenneTwister(123456)

        r1, r2, r3 = randn(rng, 5), randn(rng, 6), randn(rng, 7)
        μ, jk = randn(rng, 3), KernelCollection([r1, r2, r3])
        @test_throws ArgumentError GPCollection(randn(rng, 4), jk)
        gpc = GPCollection(μ, jk)

        @test length(gpc) == 3
        @test mean(gpc, 1) === μ[1]
        @test kernel(gpc, 1) === jk[1, 1]
        @test kernel(gpc, 1, 1) == jk[1, 1]
        @test kernel(gpc, 1, 2) == jk[1, 2]

        @test_throws ArgumentError push!(gpc, x->x, randn(3))
        r4 = randn(rng, 4)
        push!(gpc, 5.0, r4)
        @test gpc.k.k[4] == r4
        @test kernel(gpc, 1, 4) == r4[1]
        @test kernel(gpc, 2, 4) == r4[2]
        @test kernel(gpc, 3, 4) == r4[3]
        @test kernel(gpc, 4, 4) == r4[4]
    end

    # Test `append_indep!`.
    import Stheno.append_indep!
    let
        # Specification for three independent GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)

        # Make a GPCollection and append stuff to it.
        gpc = GPCollection()
        append_indep!(gpc, μ1, k1)
        append_indep!(gpc, μ2, k2)
        append_indep!(gpc, μ3, k3)

        # Check mean function vector.
        @test mean(gpc, 1) == sin
        @test mean(gpc, 2) == cos
        @test mean(gpc, 3) == tan

        # Check on-diagonal covariance functions.
        @test kernel(gpc, 1) == EQ()
        @test kernel(gpc, 2) == RQ(10.0)
        @test kernel(gpc, 3) == RQ(1.0)

        # Check off-diagonal covariance functions.
        @test kernel(gpc, 1, 2) == Constant(0.0)
        @test kernel(gpc, 1, 3) == Constant(0.0)
        @test kernel(gpc, 2, 3) == Constant(0.0)
    end

    # Check invariances for construction of a set of independent GPs.
    let
        # Specification for three independent GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)

        # Build a GPCollection.
        gpc = GPCollection()
        gp1 = GP(gpc, μ1, k1)
        gp2 = GP(gpc, μ2, k2)
        gp3 = GP(gpc, μ3, k3)

        # Check mean functions.
        @test mean(gp1) == sin
        @test mean(gp2) == cos
        @test mean(gp3) == tan

        # Check covariance functions.
        @test kernel(gp1) == EQ()
        @test kernel(gp2) == RQ(10.0)
        @test kernel(gp3) == RQ(1.0)

        # Check cross-covariance functions.
        @test kernel(gp1, gp2) == Constant(0.0)
        @test kernel(gp1, gp3) == Constant(0.0)
        @test kernel(gp2, gp3) == Constant(0.0)

        # Build a different GPCollection and make sure that an assertion is thrown
        # when trying to index into different GPCollections.
        gpc2 = GPCollection()
        gp_different = GP(gpc2, μ1, k1)
        @test_throws AssertionError kernel(gp1, gp_different)
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

        # Check that the cross-covariances havae been correctly computed.
        @test full(cov(kernel(gp1, gp_1p1), x)) ≈ 2 .* full(cov(kernel(gp1), x))
        @test full(cov(kernel(gp1, gp_1p2), x)) ≈ full(cov(kernel(gp1), x))
        @test full(cov(kernel(gp1, gp_2p1), x)) ≈ full(cov(kernel(gp1), x))
        @test full(cov(kernel(gp1, gp_2p2), x)) ≈ diagm(1e-12 * ones(N))
    end

    # Test that sampling works properly.
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

        

    end
end
