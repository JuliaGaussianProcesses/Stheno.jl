using Stheno: ConstantMean, ZeroMean, ConstantKernel, ZeroKernel, pairwise, unary_obswise,
    nobs, AV, AM

@testset "gp" begin

    # Test various utility functions.
    let

        rng, N, N′, D = MersenneTwister(123456), 4, 5, 6
        x, x′ = randn(rng, N), randn(rng, N′)
        X, X′ = randn(rng, D, N), randn(rng, D, N′)
        μ, k = ConstantMean(randn(rng)), EQ()
        kc = Stheno.LhsCross(x->sum(abs2, x), k)

        @test Stheno.permutedims(x) == x'
    end

    # Test the creation of indepenent GPs.
    let rng = MersenneTwister(123456)

        # Specification for two independent GPs.
        gpc = GPC()
        μ1, μ2 = ConstantMean(1.0), ZeroMean{Float64}()
        k1, k2 = EQ(), Linear(5)
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)

        @test mean(GP(EQ(), GPC())) == ZeroMean{Float64}()

        @test mean(f1) == μ1
        @test mean(f2) == μ2

        @test kernel(f1) == k1
        @test kernel(f2) == k2

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == ZeroKernel{Float64}()
        @test kernel(f2, f1) == ZeroKernel{Float64}()
        @test kernel(f2, f2) == k2
    end

    # Test conversion and promotion of non-GPs to GPs.
    let
        gpc = GPC()
        x, f, g = 5.0, sin, GP(ConstantMean(1.0), EQ(), gpc)
        @test convert(GP, x, gpc) == GP(ConstantMean(x), ZeroKernel{Float64}(), gpc)
        @test convert(GP, f, gpc) == GP(CustomMean(f), ZeroKernel{Float64}(), gpc)
        @test promote(x, g) == (convert(GP, x, gpc), g)
        @test promote(f, g) == (convert(GP, f, gpc), g)
    end

    # Test deterministic properties of `rand`.
    let
        rng, N, D = MersenneTwister(123456), 10, 2
        X, x, μ, k = randn(rng, D, N), randn(rng, N), ConstantMean(1), EQ()
        fX = GP(FiniteMean(μ, X), FiniteKernel(k, X), GPC())
        fx = GP(FiniteMean(μ, x), FiniteKernel(k, x), GPC())

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, fX)) == nobs(X)
        @test size(rand(rng, fX, 10)) == (nobs(X), 10)

        @test length(rand(rng, fx)) == nobs(x)
        @test size(rand(rng, fx, 10)) == (nobs(x), 10)
    end

    # Test statistical properties of `rand`.
    let
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X = randn(rng, D, N)
        μ, k = FiniteMean(ConstantMean(μ0), X), FiniteKernel(EQ(), X)
        f = GP(μ, k, GPC())

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, S)
        @test maximum(abs.(mean(f̂, 2) - AV(μ))) < 1e-2

        Σ′ = (f̂ .- AV(μ)) * (f̂ .- AV(μ))' ./ S
        @test mean(abs.(Σ′ - Matrix(cov(f)))) < 1e-2
    end

    # Test logpdf + elbo do something vaguely sensible + that elbo converges to logpdf.
    using Distributions: MvNormal, PDMat
    let
        rng, N, D, σ, gpc = MersenneTwister(123456), 5, 2, 1e-1, GPC()
        X = rand(rng, D, N)
        μ = FiniteMean(ConstantMean(1.0), X)
        kf, ky = FiniteKernel(EQ(), X), FiniteKernel(EQ() + Noise(σ^2), X)
        f = GP(μ, kf, gpc)
        y = GP(μ, ky, gpc)
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        Σ = unbox(pairwise(ky, eachindex(ky)) + 1e-6I)
        @test logpdf(y, ŷ) isa Real
        @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(AV(μ)), Σ), ŷ)

        # Ensure that the elbo is close to the logpdf when appropriate.
        @test elbo(f, ŷ, f, σ) isa Real
        @test elbo(f, ŷ, f, σ) < logpdf(y, ŷ)
        @test abs(elbo(f, ŷ, f, σ) - logpdf(y, ŷ)) < 1e-3
    end
end
