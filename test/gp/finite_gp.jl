using Stheno: FiniteGP, GPC
using Distributions: MvNormal, PDMat

@testset "finite_gp" begin

    @testset "rand (deterministic)" begin
        rng, N, D = MersenneTwister(123456), 10, 2
        X, x = ColsAreObs(randn(rng, D, N)), randn(rng, N)
        fX = FiniteGP(GP(1, EQ(), GPC()), X)
        fx = FiniteGP(GP(1, EQ(), GPC()), x)

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, fX)) == length(X)
        @test size(rand(rng, fX, 10)) == (length(X), 10)

        @test length(rand(rng, fx)) == length(x)
        @test size(rand(rng, fx, 10)) == (length(x), 10)
    end

    @testset "rand (statistical)" begin
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X = ColsAreObs(randn(rng, D, N))
        f = FiniteGP(GP(1, EQ(), GPC()), X)

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, S)
        @test maximum(abs.(mean(f̂; dims=2) - mean(f))) < 1e-2

        Σ′ = (f̂ .- mean(f)) * (f̂ .- mean(f))' ./ S
        @test mean(abs.(Σ′ - cov(f))) < 1e-2
    end

    @testset "rand (gradients)" begin
        rng, N, S = MersenneTwister(123456), 10, 3
        x = collect(range(-3.0, stop=3.0, length=N))

        # Check that the gradient w.r.t. the samples is correct.
        adjoint_test(
            x->rand(MersenneTwister(123456), FiniteGP(GP(sin, EQ(), GPC()), x), S),
            randn(rng, N, S),
            x,
        )
    end

    @testset "logpdf / elbo" begin
        rng, N, σ, gpc = MersenneTwister(123456), 10, 1e-1, GPC()
        x = collect(range(-3.0, stop=3.0, length=N))
        k_noise = OuterKernel(x->σ, Noise())
        f = FiniteGP(GP(1, EQ(), gpc), x)
        y = FiniteGP(GP(1, EQ() + k_noise, gpc), x)
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        @test logpdf(y, ŷ) isa Real
        @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(mean(f)), cov(y)), ŷ)

        # Ensure that the elbo is close to the logpdf when appropriate.
        @test elbo(f, ŷ, f, σ) isa Real
        @test abs(elbo(f, ŷ, f, σ) - logpdf(y, ŷ)) < 1e-6
    end
end
