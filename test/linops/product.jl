@testset "product" begin

    # Test the multiplication of a GP by a constant.
    let rng = MersenneTwister(123456)

        # Select some input locations.
        x, y, σ = randn(rng, 3), randn(rng, 2), exp(randn(rng))

        # Set three independent GPs.
        μ, k = sin, RQ(1.0)
        gpc = GPC()
        f, fi = GP(μ, k, gpc), GP(x->0, EQ(), gpc)
        σf = σ * f
        fσ = f * σ

        # Check that the mean has been appropriately scaled.
        @test mean(σf).(x) == σ * mean(f).(x)
        @test mean(fσ).(x) == mean(f).(x) * σ

        # Check the marginal covariance.
        @test kernel(σf).(x, y') == σ^2 .* kernel(f).(x, y')
        @test kernel(σf).(x, y') == kernel(f).(x, y') .* σ^2

        # Check the cross-covariances.
        @test kernel(σf, f).(x, y') == σ .* kernel(f).(x, y')
        @test kernel(f, σf).(x, y') == kernel(f).(x, y') .* σ
        @test kernel(fσ, f).(x, y') == σ .* kernel(f).(x, y')
        @test kernel(f, fσ).(x, y') == kernel(f).(x, y') .* σ

        # Check still independent.
        @test kernel(σf, fi).(x, y') == kernel(f, fi).(x, y')
        @test kernel(fσ, fi).(x, y') == kernel(f, fi).(x, y')
    end

    # Test the elementwise multiplication of a GP by a known function.
    let rng = MersenneTwister(123456)

        # Select some input locations.
        x, y, σ = randn(rng, 3), randn(rng, 2), exp(randn(rng))

        # Set three independent GPs.
        μ, k = sin, RQ(1.0)
        gpc = GPC()
        f, fi = GP(μ, k, gpc), GP(x->0, EQ(), gpc)
        σf = sin * f
        fσ = f * cos

        # Check that the mean has been appropriately scaled.
        @test mean(σf).(x) == sin.(x) .* mean(f).(x)
        @test mean(fσ).(x) == mean(f).(x) .* cos.(x)

        # Check the marginal covariance.
        @test kernel(σf).(x, y') == sin.(x) .* kernel(f).(x, y') .* sin.(y')
        @test kernel(fσ).(x, y') == cos.(x) .* kernel(f).(x, y') .* cos.(y')

        # Check the cross-covariances.
        @test kernel(σf, f).(x, y') == sin.(x) .* kernel(f).(x, y')
        @test kernel(f, σf).(x, y') == kernel(f).(x, y') .* sin.(y')
        @test kernel(fσ, f).(x, y') == cos.(x) .* kernel(f).(x, y')
        @test kernel(f, fσ).(x, y') == kernel(f).(x, y') .* cos.(y')

        # Check still independent.
        @test kernel(σf, fi).(x, y') == kernel(f, fi).(x, y')
        @test kernel(fσ, fi).(x, y') == kernel(f, fi).(x, y')
    end
end
