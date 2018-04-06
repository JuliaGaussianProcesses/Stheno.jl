@testset "gp" begin

    # Test the creation of indepenent GPs.
    let rng = MersenneTwister(123456)

        # Specification for two independent GPs.
        μ1, μ2 = sin, cos
        k1, k2 = EQ(), Linear(5)
        f1, f2 = GP.([μ1, μ2], [k1, k2], Ref(GPC()))

        @test mean(f1) == μ1
        @test mean(f2) == μ2

        @test kernel(f1) == k1
        @test kernel(f2) == k2

        @test kernel(f1, f1) == k1
        @test kernel(f1, f2) == Zero()
        @test kernel(f2, f1) == Zero()
        @test kernel(f2, f2) == k2
    end
end

@testset "logpdf" begin
    
end

@testset "rand" begin

    # Test deterministic features of `rand`.
    let rng = MersenneTwister(123456), N = 10, N′ = 11, D = 2
        X, X′ = randn(rng, N, D), randn(rng, N′, D)
        f = GP(X->vec(sum(sin.(X); dims=2)), EQ(), GPC())

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, f(X))) == length(X)
        @test size(rand(rng, f(X), 10)) == (length(X), 10)

        # Check that multi-GP samples have the correct dimensions.
        @test length(rand(rng, Vector{GP}([f(X), f(X′)]))[1]) == length(X)
        @test length(rand(rng, Vector{GP}([f(X), f(X′)]))[2]) == length(X′)
        @test size(rand(rng, Vector{GP}([f(X), f(X′)]), 10)[1]) == (length(X), 10)
        @test size(rand(rng, Vector{GP}([f(X), f(X′)]), 10)[2]) == (length(X′), 10)
    end

    # # Test some statistical properties of `rand`.
    # let rng = MersenneTwister(123456)
    #     x, x′ = randn(rng, 10), randn(rng, 11)
    #     f = GP(CustomMean(sin), EQ(), GPC())

    #     # Check mean + covariance estimates approximately converge for single-GP sampling.
    #     S = 100000
    #     f̂ = rand(rng, f(x), S)

    #     μ̂, μ = mean(f̂, dims=2), sin.(x)
    #     @test mean(abs.(μ̂ .- μ)) < 1e-2

    #     Σ̂, Σ = (f̂ .- sin.(x)) * (f̂ .- sin.(x))' ./ S, cov(kernel(f(x)))
    #     @test mean(abs.(Σ̂ .- Σ)) < 1e-2
    
    #     # Check mean + covariance estimates approximately converge for multi-GP sampling.
    #     S, x̂ = 100000, vcat(x, x′)
    #     f̂ = rand(rng, Vector{GP}([f(x), f(x′)]), S)
    #     f̂ = vcat(f̂[1], f̂[2])

    #     μ̂, μ = mean(f̂, dims=2), sin.(x̂)
    #     @test mean(abs.(μ̂ .- μ)) < 1e-2

    #     Σ̂, Σ = (f̂ .- sin.(x̂)) * (f̂ .- sin.(x̂))' ./ S, cov(kernel(f(x̂)))
    #     @test mean(abs.(Σ̂ .- Σ)) < 1e-2
    # end
end
