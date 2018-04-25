using Stheno: LhsFiniteCrossKernel, RhsFiniteCrossKernel

function linops_tests()
    @testset "lin_ops" begin
        lin_ops_indexing_tests()
        lin_ops_conditioning_tests()
    end
end

function lin_ops_indexing_tests()

    # Set up some GPs.
    rng, N, N′, D, gpc = MersenneTwister(123456), 4, 5, 2, GPC()
    X, X′ = randn(rng, N, D), randn(rng, N′, D)
    μ1, μ2 = ConstantMean(1), ConstantMean(-1.5)
    k1, k2 = EQ(), Linear(0.5)
    f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
    f3 = f1(X)

    # Check mean and marginal covariance under indexing.
    @test mean(f3) == FiniteMean(μ1, X)
    @test kernel(f3) == FiniteKernel(k1, X)
    @test kernel(f3, f1) == LhsFiniteCrossKernel(kernel(f1), X)
    @test kernel(f1, f3) == RhsFiniteCrossKernel(kernel(f1), X)
    @test kernel(f3, f2) == LhsFiniteCrossKernel(ZeroKernel{Float64}(), X)
    @test kernel(f2, f3) == RhsFiniteCrossKernel(ZeroKernel{Float64}(), X)

    # Check that kernel types are correct.
    f4 = f1(X′)
    @test typeof(kernel(f4)) <: FiniteKernel
    @test typeof(kernel(f4, f4)) <: FiniteKernel
    @test typeof(kernel(f4, f3)) <: LhsFiniteCrossKernel
    @test typeof(kernel(f3, f4)) <: RhsFiniteCrossKernel
    @test typeof(kernel(f4, f1)) <: LhsFiniteCrossKernel
    @test typeof(kernel(f1, f4)) <: RhsFiniteCrossKernel
    @test typeof(kernel(f2, f4)) <: RhsFiniteCrossKernel
    @test typeof(kernel(f4, f2)) <: LhsFiniteCrossKernel

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

function lin_ops_conditioning_tests()
    rng, N, N′, D = MersenneTwister(123456), 5, 6,  2
    X, X′ = randn(rng, N, D), randn(rng, N′, D)
    y = randn(rng, N)

    # Test mechanics for finite conditioned process with single conditioning.
    f = GP(ConstantMean(1), EQ(), GPC())
    f′ = f | (f(X) ← y)
    @test length(f′) == Inf
    @test length(rand(rng, f′, X)) == N
    @test maximum(rand(rng, f′, X) - y) < 1e-5
    @test mean(f′, X) ≈ y
    @test all(abs.(Matrix(cov(kernel(f′), X))) .< 1e-9)
end
