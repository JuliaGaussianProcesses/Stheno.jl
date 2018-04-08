@testset "gp" begin

    # Test the creation of indepenent GPs.
    let rng = MersenneTwister(123456)

        # Specification for two independent GPs.
        μ1, μ2 = ConstantMean(1.0), ZeroMean{Float64}()
        k1, k2 = EQ(), Linear(5)
        f1, f2 = GP.([μ1, μ2], [k1, k2], Ref(GPC()))

        @test mean_function(f1) == μ1
        @test mean_function(f2) == μ2

        @test kernel(f1) == k1
        @test kernel(f2) == k2

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == ZeroKernel{Float64}()
        @test kernel(f2, f1) == ZeroKernel{Float64}()
        @test kernel(f2, f2) == k2
    end
end

@testset "logpdf" begin
    
end

@testset "rand" begin

    # Test deterministic properties of `rand`.
    let
        rng, N, N′, D = MersenneTwister(123456), 10, 11, 2
        X, X′, gpc = randn(rng, N, D), randn(rng, N′, D), GPC()
        μ, k = FiniteMean(ConstantMean(1), X), FiniteKernel(EQ(), randn(rng, N, D))
        μ′, k′ = FiniteMean(ConstantMean(1), X′), FiniteKernel(EQ(), randn(rng, N′, D))
        f, f′ = GP(μ, k, gpc), GP(μ′, k′, gpc)

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, f)) == size(X, 1)
        @test size(rand(rng, f, 10)) == (size(X, 1), 10)
    end

    # Test statistical properties of `rand`.
    let
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X = randn(rng, N, D)
        μ, k = FiniteMean(ConstantMean(μ0), X), FiniteKernel(EQ(), X)
        f = GP(μ, k, GPC())

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, S)
        @test maximum(abs.(mean(f̂, dims=2) - mean(μ))) < 1e-2

        Σ′ = (f̂ .- mean(μ)) * (f̂ .- mean(μ))' ./ S
        @test mean(abs.(Σ′ - Matix(cov(f)))) < 1e-2
    end
end
