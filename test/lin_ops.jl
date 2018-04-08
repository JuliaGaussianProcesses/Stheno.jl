using Stheno: LhsFiniteCrossKernel, RhsFiniteCrossKernel

@testset "lin_ops" begin

    # Test indexing into a GP and that the cross-covariances with another independent GP
    # are zero.
    let rng = MersenneTwister(123456)

        # Set up some GPs.
        rng, N, N′, D, gpc = MersenneTwister(123456), 4, 5, 2, GPC()
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        μ1, μ2 = ConstantMean(1), ConstantMean(-1.5)
        k1, k2 = EQ(), Linear(0.5)
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
        f3 = f1(X)

        # Check mean and marginal covariance under indexing.
        @test mean(f3) == mean(f1, X)
        @test cov(f3) == cov(f1, X)
        @test kernel(f3, f1) == LhsFiniteCrossKernel(kernel(f1), X)
        @test kernel(f1, f3) == RhsFiniteCrossKernel(kernel(f1), X)
        @test kernel(f3, f2) == LhsFiniteCrossKernel(ZeroKernel{Float64}(), X)
        @test kernel(f2, f3) == RhsFiniteCrossKernel(ZeroKernel{Float64}(), X)

        # Check that kernel types are correct.
        f4 = f1(X′)
        @test typeof(kernel(f4)) <: FiniteKernel
        @test typeof(kernel(f4, f4)) <: FiniteKernel
        @test typeof(kernel(f4, f3)) <: FiniteCrossKernel
        @test typeof(kernel(f3, f4)) <: FiniteCrossKernel
        @test typeof(kernel(f4, f1)) <: LhsFiniteCrossKernel
        @test typeof(kernel(f1, f4)) <: RhsFiniteCrossKernel
        @test typeof(kernel(f2, f4)) <: RhsFiniteCrossKernel
        @test typeof(kernel(f4, f2)) <: LhsFiniteCrossKernel

        # Check that kernels evaluate correctly.
        @test mean(f4) == mean(f1, X′)
        @test cov(f4) == cov(f1, X′)
        @test xcov(f3, f4) == xcov(f4, f3)'
        @test xcov(f3, f4) == xcov(f1, X, X′)
        @test xcov(f4, f3) == xcov(f1, X′, X)

        # These tests are currently unnecessary as nested indexing has been removed.
    #     # Check that nested indexing works as expected.
    #     f5 = f3(1:2)
    #     @test typeof(kernel(f5)) <: Finite
    #     @test typeof(kernel(f4, f5)) <: Finite
    #     @test typeof(kernel(f5, f4)) <: Finite
    #     @test typeof(kernel(f5, f3)) <: Finite
    #     @test typeof(kernel(f3, f5)) <: Finite
    #     @test typeof(kernel(f5, f2)) <: LhsFinite
    #     @test typeof(kernel(f2, f5)) <: RhsFinite
    #     @test typeof(kernel(f5, f1)) <: LhsFinite
    #     @test typeof(kernel(f1, f5)) <: RhsFinite

    #     # Check that the kernels evaluate correctly.
    #     id5, id4, id3 = eachindex(f5), eachindex(f4), eachindex(f3)
    #     @test kernel(f5).(id5, id5') == kernel(f3).(id5, id5')
    #     @test kernel(f5, f4).(id5, id4') == kernel(f1).(x[id5], y[id4]')
    #     @test kernel(f4, f5).(id4, id5') == kernel(f5, f4).(id5, id4')'
    #     @test kernel(f5, f3).(id5, id3') == kernel(f1).(x[id5], x[id3]')
    #     @test kernel(f3, f5).(id3, id5') == kernel(f5, f3).(id5, id3')'
    end

    # Test inference.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6,  2
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        y = randn(rng, N)

        # Test mechanics for finite conditioned process with single conditioning.
        f = GP(ConstantMean(1), EQ(), GPC())
        f′X = f(X) | (f(X) ← y)
        f′X′ = f(X′) | (f(X) ← y)
        @test length(f′X) == N && length(rand(rng, f′X)) == N
        @test length(f′X′) == N′ && length(rand(rng, f′X′)) == N′

        # Test mechanics for infinite conditioned process with single conditioning.
        f = GP(ConstantMean(1), EQ(), GPC())
        # f′ = f | (f(X) ← y)
        # @test !isfinite(f′)


        # f′x = f(x) | (f(x) ← f̂)
        # f′x′ = f(x′) | (f(x) ← f̂)
        # f′ = f | (f(x) ← f̂)

        # # Test finite GP posterior.
        # idx = collect(eachindex(f′x))
        # @test dims(f′x) == length(x)
        # @test mean(f′x).(idx) ≈ f̂
        # @test all(kernel(f′x).(idx, idx') .- diagm(0 => 2e-9 * fill!(similar(x), 1)) .< 1e-12)
        # @test dims(f′x′) == length(x′)

        # # Test process posterior works.
        # @test mean(f′).(x) ≈ f̂
        # @test all(kernel(f′).(x, x') .- diagm(0 => 2e-9 * fill!(similar(x), 1)) .< 1e-12)

        # # Test that covariances are computed properly.
        # @test maximum(abs.(Matrix(cov(f′x)) .- 2 .* kernel(f′).(x, x'))) < 1e-12
        # @test Matrix(cov(f′x′)) ≈ kernel(f′).(x′, x′')
    end
end
