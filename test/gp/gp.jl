using Stheno: OuterKernel, BinaryKernel

@testset "gp" begin

    # # Test various utility functions.
    # let

    #     rng, N, N′, D = MersenneTwister(123456), 4, 5, 6
    #     x, x′ = DataSet(randn(rng, N)), DataSet(randn(rng, N′))
    #     X, X′ = DataSet(randn(rng, D, N)), DataSet(randn(rng, D, N′))
    #     μ, k = ConstantMean(randn(rng)), EQ()
    #     kc = Stheno.LhsCross(x->sum(abs2, x), k)

    #     @test Stheno.permutedims(x) == x'
    # end

    # Check various ways to construct a GP do what you would expect.
    let
        m = 5.1
        @test mean(GP(EQ(), GPC())) isa ZeroMean
        @test mean(GP(zero(m), EQ(), GPC())) isa ZeroMean
        @test mean(GP(one(m), EQ(), GPC())) isa OneMean

        x = randn(10)
        @test map(mean(GP(sin, EQ(), GPC())), x) == sin.(x)
    end

    # Test the creation of indepenent GPs.
    let
        rng = MersenneTwister(123456)

        # Specification for two independent GPs.
        gpc = GPC()
        μ1, μ2 = OneMean(), ZeroMean()
        k1, k2 = EQ(), Linear()
        f1, f2 = GP(μ1, k1, gpc), GP(μ2, k2, gpc)

        @test mean(f1) == μ1
        @test mean(f2) == μ2

        @test kernel(f1) == k1
        @test kernel(f2) == k2

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == ZeroKernel()
        @test kernel(f2, f1) == ZeroKernel()
        @test kernel(f2, f2) == k2
    end

    # # Test conversion and promotion of non-GPs to GPs.
    # let
    #     gpc = GPC()
    #     x, f, g = 5.0, sin, GP(OneMean(), EQ(), gpc)
    #     @test convert(GP, x, gpc) == GP(OneMean(), ZeroKernel(), gpc)
    #     @test convert(GP, f, gpc) == GP(CustomMean(f), ZeroKernel(), gpc)
    #     @test promote(x, g) == (convert(GP, x, gpc), g)
    #     @test promote(f, g) == (convert(GP, f, gpc), g)
    # end

    # Test deterministic properties of `rand`.
    let
        rng, N, D = MersenneTwister(123456), 10, 2
        X, x, μ, k = ColsAreObs(randn(rng, D, N)), randn(rng, N), OneMean(), EQ()
        fX = GP(FiniteMean(μ, X), FiniteKernel(k, X), GPC())
        fx = GP(FiniteMean(μ, x), FiniteKernel(k, x), GPC())

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, fX)) == length(X)
        @test size(rand(rng, fX, 10)) == (length(X), 10)

        @test length(rand(rng, fx)) == length(x)
        @test size(rand(rng, fx, 10)) == (length(x), 10)
    end

    # Test statistical properties of `rand`.
    let
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X = ColsAreObs(randn(rng, D, N))
        μ, k = FiniteMean(OneMean(), X), FiniteKernel(EQ(), X)
        f = GP(μ, k, GPC())

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, S)
        @test maximum(abs.(mean(f̂; dims=2) - mean_vec(f))) < 1e-2

        Σ′ = (f̂ .- mean_vec(f)) * (f̂ .- mean_vec(f))' ./ S
        @test mean(abs.(Σ′ - cov(f))) < 1e-2
    end

    # Test `rand` gradients.
    let
        rng, N, S = MersenneTwister(123456), 10, 3
        x = collect(range(-3.0, stop=3.0, length=N))

        # Check that the gradient w.r.t. the samples is correct.
        adjoint_test(
            x->begin
                f = GP(FiniteMean(CustomMean(sin), x), FiniteKernel(EQ(), x), GPC())
                return rand(MersenneTwister(123456), f, S)
            end,
            randn(rng, N, S),
            x,
        )
    end

    # Test logpdf + elbo do something vaguely sensible + that elbo converges to logpdf.
    using Distributions: MvNormal, PDMat
    let
        rng, N, σ, gpc = MersenneTwister(123456), 10, 1e-1, GPC()
        x = collect(range(-3.0, stop=3.0, length=N))
        μ = FiniteMean(OneMean(), x)
        kf = FiniteKernel(EQ(), x)
        ky = FiniteKernel(BinaryKernel(+, EQ(), OuterKernel(x->σ, Noise())), x)
        f = GP(μ, kf, gpc)
        y = GP(μ, ky, gpc)
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        Σ = pairwise(ky, :)
        @test logpdf(y, ŷ) isa Real
        @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(mean_vec(f)), Σ), ŷ)

        # Ensure that the elbo is close to the logpdf when appropriate.
        @test elbo(f, ŷ, f, σ) isa Real
        @test abs(elbo(f, ŷ, f, σ) - logpdf(y, ŷ)) < 1e-6
    end
end
