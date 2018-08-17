using Random, LinearAlgebra
using Stheno: FiniteKernel, FiniteCrossKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    LhsFiniteZeroCrossKernel, RhsFiniteZeroCrossKernel

@testset "indexing" begin

let
    # Set up some GPs.
    rng, N, N′, D, gpc = MersenneTwister(123456), 4, 5, 2, GPC()
    X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    μ1, μ2 = ConstantMean(1), ConstantMean(-1.5)
    k1, k2 = EQ(), Linear(0.5)
    f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
    f3 = f1(X)

    # Check mean and marginal covariance under indexing.
    @test mean(f3) == FiniteMean(μ1, X)
    @test kernel(f3) == FiniteKernel(k1, X)
    @test kernel(f3, f1) == LhsFiniteCrossKernel(kernel(f1), X)
    @test kernel(f1, f3) == RhsFiniteCrossKernel(kernel(f1), X)
    @test kernel(f3, f2) == LhsFiniteZeroCrossKernel(X)
    @test kernel(f2, f3) == RhsFiniteZeroCrossKernel(X)

    # Check that kernel types are correct.
    f4 = f1(X′)
    @test kernel(f4) isa FiniteKernel
    @test kernel(f4, f4) isa FiniteKernel
    @test kernel(f4, f3) isa FiniteCrossKernel
    @test kernel(f3, f4) isa FiniteCrossKernel
    @test kernel(f4, f1) isa LhsFiniteCrossKernel
    @test kernel(f1, f4) isa RhsFiniteCrossKernel
    @test kernel(f2, f4) isa RhsFiniteZeroCrossKernel
    @test kernel(f4, f2) isa LhsFiniteZeroCrossKernel

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

let
    # Set up some GPs.
    rng, N, N′, D, gpc = MersenneTwister(123456), 4, 5, 2, GPC()
    X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
    μ1, μ2 = ConstantMean(1.0), ConstantMean(-1.5)
    k1, k2 = EQ(), Linear(0.5)
    f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)
    f3, f4 = f1(X), f2(X′)
    fj = BlockGP([f1, f2])
    fjXX′ = fj(BlockData([X, X′]))

    @test fjXX′ isa BlockGP

    Σ11, Σ12, Σ22 = xcov(f3, f3), xcov(f3, f4), xcov(f4, f4)
    Σ_manual = BlockMatrix(reshape([Σ11, Σ12', Σ12, Σ22], 2, 2))

    # Test that indexing a BlockGP is consistent with separate indexing.
    @test mean_vec(fjXX′) == vcat(mean_vec(f3), mean_vec(f4))
    @test cov(fjXX′) == Σ_manual
    @test xcov(fjXX′, fjXX′) == cov(fjXX′)
    @test xcov(fjXX′, f3) == BlockMatrix(reshape([xcov(f3, f3), xcov(f4, f3)], 2, 1))

    # Check that broadcasting functionality works appropriately.
    @test cov(BlockGP([f1, f2])(X)) == cov(BlockGP([f1(X), f2(X)]))
    @test cov(BlockGP([f1, f2])(X)(1:N-1)) == cov(BlockGP([f1(X), f2(X)])(1:N-1))
end

end # @testset indexing
