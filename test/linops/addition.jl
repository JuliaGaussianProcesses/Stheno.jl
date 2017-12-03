@testset "addition" begin

    # Test z + GP and GP + z.
    let rng = MersenneTwister(123456)

        # Select some input locations.
        x, y, σ = randn(rng, 3), randn(rng, 2), exp(randn(rng))

        # Set three independent GPs.
        μ, k = sin, RQ(1.0)
        gpc = GPC()
        f, fi = GP(μ, k, gpc), GP(x->0, EQ(), gpc)
        σf = σ + f
        fσ = f + σ

        # Check that the mean has been appropriately scaled.
        @test mean(σf).(x) == σ .+ mean(f).(x)
        @test mean(fσ).(x) == mean(f).(x) .+ σ

        # Check the marginal covariance.
        @test kernel(σf).(x, y') == kernel(f).(x, y')
        @test kernel(σf).(x, y') == kernel(f).(x, y')

        # Check the cross-covariances.
        @test kernel(σf, f).(x, y') == kernel(f).(x, y')
        @test kernel(f, σf).(x, y') == kernel(f).(x, y')
        @test kernel(fσ, f).(x, y') == kernel(f).(x, y')
        @test kernel(f, fσ).(x, y') == kernel(f).(x, y')

        # Check still independent.
        @test kernel(σf, fi).(x, y') == kernel(f, fi).(x, y')
        @test kernel(fσ, fi).(x, y') == kernel(f, fi).(x, y')
    end

    # Test g(x) + GP(x) and GP(x) + g(x).
    let rng = MersenneTwister(123456)

        # Select some input locations.
        x, y, σ = randn(rng, 3), randn(rng, 2), exp(randn(rng))

        # Set three independent GPs.
        μ, k = sin, RQ(1.0)
        gpc = GPC()
        f, fi = GP(μ, k, gpc), GP(x->0, EQ(), gpc)
        σf = sin + f
        fσ = f + cos

        # Check that the mean has been appropriately scaled.
        @test mean(σf).(x) == sin.(x) .+ mean(f).(x)
        @test mean(fσ).(x) == mean(f).(x) .+ cos.(x)

        # Check the marginal covariance.
        @test kernel(σf).(x, y') == kernel(f).(x, y')
        @test kernel(σf).(x, y') == kernel(f).(x, y')

        # Check the cross-covariances.
        @test kernel(σf, f).(x, y') == kernel(f).(x, y')
        @test kernel(f, σf).(x, y') == kernel(f).(x, y')
        @test kernel(fσ, f).(x, y') == kernel(f).(x, y')
        @test kernel(f, fσ).(x, y') == kernel(f).(x, y')

        # Check still independent.
        @test kernel(σf, fi).(x, y') == kernel(f, fi).(x, y')
        @test kernel(fσ, fi).(x, y') == kernel(f, fi).(x, y')
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
        @test kernel(f_1p1).(x, x') ≈ 4 .* k1.(x, x')
        @test kernel(f_1p2).(x, x') ≈ k1.(x, x') .+ k2.(x, x')
        @test kernel(f_2p1).(x, x') ≈ k2.(x, x') .+ k1.(x, x')
        @test kernel(f_2p2).(x, x') ≈ 4 .* k2.(x, x')

        # Check that the cross-covariances have been correctly computed.
        @test kernel(f1, f_1p1).(x, x') ≈ 2 .* kernel(f1).(x, x')
        @test kernel(f1, f_1p2).(x, x') ≈ kernel(f1).(x, x')
        @test kernel(f1, f_2p1).(x, x') ≈ kernel(f1).(x, x')
        @test kernel(f1, f_2p2).(x, x') ≈ zeros(3, 3)

        # Check that the cross-covariances match.
        @test kernel(f1, f_1p1).(x, x') == permutedims(kernel(f_1p1, f1).(x, x'), [2, 1])
        @test kernel(f1, f_1p2).(x, x') == permutedims(kernel(f_1p2, f1).(x, x'), [2, 1])
        @test kernel(f1, f_2p1).(x, x') == permutedims(kernel(f_2p1, f1).(x, x'), [2, 1])
        @test kernel(f1, f_2p2).(x, x') == permutedims(kernel(f_2p2, f1).(x, x'), [2, 1])

        if check_mem

            # Memory performance tests.
            @test memory(@benchmark $(mean(f_1p1))(1.0) seconds=0.1) == 0
            @test memory(@benchmark $(mean(f_1p2))(0.0) seconds=0.1) == 0
            @test memory(@benchmark $(mean(f_2p1))(-1.0) seconds=0.1) == 0
            @test memory(@benchmark $(mean(f_2p2))(5.0) seconds=0.1) == 0
            @test memory(@benchmark $(kernel(f1, f_1p1))(1.0, 0.0) seconds=0.1) == 0
            @test memory(@benchmark $(kernel(f_2p2))(0.0, 1.0) seconds=0.1) == 0
        end
    end
end
