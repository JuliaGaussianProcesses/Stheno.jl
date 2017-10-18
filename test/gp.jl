@testset "gp" begin

    # Test the creation of indepenent GPs.
    let rng = MersenneTwister(123456)

        # Specification for three independent GPs.
        μ1, μ2, μ3 = sin, cos, tan
        k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
        f1, f2, f3 = GP.([μ1, μ2, μ3], [k1, k2, k3], GPC())

        @test mean(f1) == μ1
        @test mean(f2) == μ2
        @test mean(f3) == μ3

        @test kernel(f1) == k1
        @test kernel(f2) == k2
        @test kernel(f3) == k3

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == Constant(0.0)
        @test kernel(f1, f3) == Constant(0.0)
        @test kernel(f2, f1) == Constant(0.0)
        @test kernel(f2, f2) == k2
        @test kernel(f2, f3) == Constant(0.0)
        @test kernel(f3, f1) == Constant(0.0)
        @test kernel(f3, f2) == Constant(0.0)
        @test kernel(f3, f3) == k3
    end

    # # Test `append_indep!`.
    # import Stheno.append_indep!
    # let
    #     # Specification for three independent GPs.
    #     μ1, μ2, μ3 = sin, cos, tan
    #     k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)

    #     # Make a GPC and append stuff to it.
    #     gpc = GPC()
    #     append_indep!(gpc, μ1, k1)
    #     append_indep!(gpc, μ2, k2)
    #     append_indep!(gpc, μ3, k3)

    #     # Check mean function vector.
    #     @test mean(gpc, 1) == sin
    #     @test mean(gpc, 2) == cos
    #     @test mean(gpc, 3) == tan

    #     # Check on-diagonal covariance functions.
    #     @test kernel(gpc, 1) == EQ()
    #     @test kernel(gpc, 2) == RQ(10.0)
    #     @test kernel(gpc, 3) == RQ(1.0)

    #     # Check off-diagonal covariance functions.
    #     @test kernel(gpc, 1, 2) == Constant(0.0)
    #     @test kernel(gpc, 1, 3) == Constant(0.0)
    #     @test kernel(gpc, 2, 3) == Constant(0.0)
    # end

    # # Check invariances for construction of a set of independent GPs.
    # let
    #     # Specification for three independent GPs.
    #     μ1, μ2, μ3 = sin, cos, tan
    #     k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)

    #     # Build a GPC.
    #     gpc = GPC()
    #     gp1 = GP(gpc, μ1, k1)
    #     gp2 = GP(gpc, μ2, k2)
    #     gp3 = GP(gpc, μ3, k3)

    #     # Check mean functions.
    #     @test mean(gp1) == sin
    #     @test mean(gp2) == cos
    #     @test mean(gp3) == tan

    #     # Check covariance functions.
    #     @test kernel(gp1) == EQ()
    #     @test kernel(gp2) == RQ(10.0)
    #     @test kernel(gp3) == RQ(1.0)

    #     # Check cross-covariance functions.
    #     @test kernel(gp1, gp2) == Constant(0.0)
    #     @test kernel(gp1, gp3) == Constant(0.0)
    #     @test kernel(gp2, gp3) == Constant(0.0)

    #     # Build a different GPC and make sure that an assertion is thrown
    #     # when trying to index into different GPCs.
    #     gpc2 = GPC()
    #     gp_different = GP(gpc2, μ1, k1)
    #     @test_throws AssertionError kernel(gp1, gp_different)
    # end

    # # Test concatenation of GP means.
    # let rng = MersenneTwister(123456)

    #     # Select some input locations.
    #     N1, N2, N3 = 2, 3, 4
    #     x1, x2, x3 = randn.(rng, [N1, N2, N3])

    #     # Set up some GPs.
    #     μ1, μ2, μ3 = sin, cos, tan
    #     k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
    #     f1, f2, f3 = GP.(GPC(), [μ1, μ2, μ3], [k1, k2, k3])

    #     # Test that the mean functions work correctly for observations of a single GP.
    #     @test mean(GPInputSetPair(f1, x1)) == μ1.(x1)
    #     @test mean(GPInputSetPair(f2, x2)) == μ2.(x2)
    #     @test mean(GPInputSetPair(f3, x3)) == μ3.(x3)

    #     # Test the that mean functions work correctly for collections of GPs.
    #     @test mean([GPInputSetPair(f1, x1)])
    # end

    # # Test that sampling works properly.
    # let rng = MersenneTwister(123456)

    #     # Select some input locations.
    #     N = 2
    #     x = randn(rng, N)

    #     # Set up a pair of GPs.
    #     μ1, μ2, μ3 = sin, cos, tan
    #     k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
    #     gpc = GPC()
    #     gp1 = GP(gpc, μ1, k1)
    #     gp2 = GP(gpc, μ2, k2)
    #     gp3 = GP(gpc, μ3, k3)
    # end

    # # Test that certain aspects of posterior prediction work correctly in a variety of
    # # situations - single GP, some combinations of GPs.
    # let rng = MersenneTwister(123456)

    #     # Set up some GPs.
    #     μ1, μ2, μ3 = sin, cos, tan
    #     k1, k2, k3 = EQ(), RQ(10.0), RQ(1.0)
    #     gp1, gp2, gp3 = GP.(GPC(), [μ1, μ2, μ3], [k1, k2, k3])

    #     x, xs = randn.(rng, [4, 5])
    #     # @test predict([(gp1, x)]) ≈ zeros(size(x))
    # end
end
