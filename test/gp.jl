@testset "gp" begin

    # Test the creation of indepenent GPs.
    let rng = MersenneTwister(123456)

        # Specification for two independent GPs.
        μ1, μ2 = ConstantMean(1.0), ZeroMean{Float64}()
        k1, k2 = EQ(), Linear(5)
        f1, f2 = GP.([μ1, μ2], [k1, k2], Ref(GPC()))

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
        X, μ, k = randn(rng, N, D), ConstantMean(1), EQ()
        f = GP(μ, k, GPC())

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, f, X)) == size(X, 1)
        @test size(rand(rng, f, X, 10)) == (size(X, 1), 10)
    end

    # Test statistical properties of `rand`.
    let
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X, μ, k = randn(rng, N, D), ConstantMean(μ0), EQ()
        f = GP(μ, k, GPC())

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, X, S)
        @test maximum(abs.(mean(f̂, dims=2) - mean(μ, X))) < 1e-2

        Σ′ = (f̂ .- mean(μ, X)) * (f̂ .- mean(μ, X))' ./ S
        @test mean(abs.(Σ′ - Matrix(cov(f, X)))) < 1e-2
    end

    # Test logpdf + elbo do something vaguely sensible + that elbo converges to logpdf.
    let
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        f = GP(ConstantMean(1.0), EQ(), GPC())
        y, y′ = rand(rng, f, X), rand(rng, f, X′)
        @test logpdf([f], [X], BlockVector([y])) isa Real
        @test logpdf(f, X, y) == logpdf([f], [X], BlockVector([y]))
        @test logpdf([f, f], [X, X′], BlockVector([y, y′])) isa Real

        @show logpdf(f, X, y)
        @show elbo([f], [X], BlockVector([y]), [f], [X], 1e-12)
    end
end
