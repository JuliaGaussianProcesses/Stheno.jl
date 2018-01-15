@testset "lin_ops" begin

    # Test mean_vector.
    let rng = MersenneTwister(123456)
        x, x′ = randn(rng, 5), randn(rng, 10)
        f = GP(CustomMean(sin), EQ(), GPC())
        @test mean_vector(f(x)) == sin.(x)
        @test mean_vector([f(x), f(x′)]) == vcat(sin.(x), sin.(x′))
    end

    # Test indexing into a GP and that the cross-covariances with another independent GP
    # are zero.
    let rng = MersenneTwister(123456)

        # Set up some GPs.
        x, y = randn(rng, 3), randn(rng, 2)
        μ1, μ2 = CustomMean(sin), CustomMean(cos)
        k1, k2 = EQ(), RQ(10.0)
        gpc = GPC()
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
        f3 = f1(x)

        # Check mean and marginal covariance under indexing.
        idx, idy = eachindex(x), eachindex(y)
        @test mean(f3).(idx) == mean(f1).(x)
        @test kernel(f3).(idx, idx') == kernel(f1).(x, Transpose(x))
        @test kernel(f3, f1).(idx, y') == kernel(f1).(x, Transpose(y))
        @test kernel(f1, f3).(y, idx') == kernel(f1).(y, Transpose(x))
        @test all(kernel(f3, f2).(idx, y') .== 0.0)
        @test all(kernel(f2, f3).(y', idx) .== 0.0)

        if check_mem
            @test memory(@benchmark $(mean(f3))(1) seconds=0.1) == 0
            @test memory(@benchmark $(kernel(f3))(1, 2) seconds=0.1) == 0
        end

        # Check that kernel types are correct.
        f4 = f1(y)
        @test typeof(kernel(f4)) <: Finite
        @test typeof(kernel(f4, f4)) <: Finite
        @test typeof(kernel(f4, f3)) <: Finite
        @test typeof(kernel(f3, f4)) <: Finite
        @test typeof(kernel(f4, f1)) <: LhsFinite
        @test typeof(kernel(f1, f4)) <: RhsFinite
        @test typeof(kernel(f2, f4)) <: RhsFinite
        @test typeof(kernel(f4, f2)) <: LhsFinite

        # Check that kernels evaluate correctly.
        @test kernel(f4).(idy, idy') == kernel(f1).(y, y')
        @test kernel(f3, f4).(idx, idy') == kernel(f1).(x, y')
        @test kernel(f4, f3).(idy, idx') == kernel(f3, f4).(idx, idy')'

        # Check that nested indexing works as expected.
        f5 = f3(1:2)
        @test typeof(kernel(f5)) <: Finite
        @test typeof(kernel(f4, f5)) <: Finite
        @test typeof(kernel(f5, f4)) <: Finite
        @test typeof(kernel(f5, f3)) <: Finite
        @test typeof(kernel(f3, f5)) <: Finite
        @test typeof(kernel(f5, f2)) <: LhsFinite
        @test typeof(kernel(f2, f5)) <: RhsFinite
        @test typeof(kernel(f5, f1)) <: LhsFinite
        @test typeof(kernel(f1, f5)) <: RhsFinite

        # Check that the kernels evaluate correctly.
        id5, id4, id3 = eachindex(f5), eachindex(f4), eachindex(f3)
        @test kernel(f5).(id5, id5') == kernel(f3).(id5, id5')
        @test kernel(f5, f4).(id5, id4') == kernel(f1).(x[id5], y[id4]')
        @test kernel(f4, f5).(id4, id5') == kernel(f5, f4).(id5, id4')'
        @test kernel(f5, f3).(id5, id3') == kernel(f1).(x[id5], x[id3]')
        @test kernel(f3, f5).(id3, id5') == kernel(f5, f3).(id5, id3')'
    end

    # Test inference.
    let rng = MersenneTwister(123456)
        x, x′, f̂ = randn(rng, 3), randn(rng, 2), randn(rng, 3)

        f = GP(CustomMean(sin), EQ(), GPC())
        f′x = f(x) | (f(x) ← f̂)
        f′x′ = f(x′) | (f(x) ← f̂)
        f′ = f | (f(x) ← f̂)

        # Test finite GP posterior.
        idx = collect(eachindex(f′x))
        @test dims(f′x) == length(x)
        @test mean(f′x).(idx) ≈ f̂
        @test all(kernel(f′x).(idx, idx') .- diagm(0 => 2e-9 * fill!(similar(x), 1)) .< 1e-12)
        @test dims(f′x′) == length(x′)

        # Test process posterior works.
        @test mean(f′).(x) ≈ f̂
        @test all(kernel(f′).(x, x') .- diagm(0 => 2e-9 * fill!(similar(x), 1)) .< 1e-12)

        # Test that covariances are computed properly.
        @test maximum(abs.(full(cov(f′x)) .- 2 .* kernel(f′).(x, x'))) < 1e-12
        @test full(cov(f′x′)) ≈ kernel(f′).(x′, x′')
    end
end
