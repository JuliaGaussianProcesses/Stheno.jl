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
end
