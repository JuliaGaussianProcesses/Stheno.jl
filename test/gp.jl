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

    # Test a generic toy problem.
    let
        rng = MersenneTwister(123456)
        N, S = 5, 100000
        μ_vec, x = randn(rng, N), randn(rng, N)
        μ, k = n::Int->μ_vec[n], (m::Int, n::Int)->EQ()(x[m], x[n])
        d = Normal(nothing, nothing, μ, k, N, GPC())

        @test mean(d) == μ
        @test mean(d).(1:N) == μ_vec
        @test dims(d) == N
        for m in 1:N, n in 1:N
            @test kernel(d)(m, n) == EQ()(x[m], x[n])
        end

        x̂ = sample(rng, d, S)
        @test size(x̂) == (N, S)
        @test maximum(abs.(mean(x̂, 2) - mean(d).(1:N))) < 1e-2
        Σ = broadcast(kernel(d), collect(1:N), RowVector(collect(1:N)))
        @test maximum(abs.(cov(x̂, 2) - Σ)) < 1e-2

        f = randn(rng, N)
        condition!(d, f)
        @test d in keys(d.gpc.obs)
        @test f in values(d.gpc.obs)
        @test d.gpc.obs[d] === f
    end

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
