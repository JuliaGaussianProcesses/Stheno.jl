@testset "sample" begin

    # Test deterministic features of `sample`.
    let rng = MersenneTwister(123456)
        x, x′ = randn(rng, 10), randn(rng, 11)
        f = GP(sin, EQ(), GPC())

        # Check that single-GP samples have the correct dimensions.
        @test length(sample(rng, f(x))) == length(x)
        @test size(sample(rng, f(x), 10)) == (length(x), 10)

        # Check that multi-GP samples have the correct dimensions.
        @test length(sample(rng, Vector{GP}([f(x), f(x′)]))[1]) == length(x)
        @test length(sample(rng, Vector{GP}([f(x), f(x′)]))[2]) == length(x′)
        @test size(sample(rng, Vector{GP}([f(x), f(x′)]), 10)[1]) == (length(x), 10)
        @test size(sample(rng, Vector{GP}([f(x), f(x′)]), 10)[2]) == (length(x′), 10)
    end

    # Test some statistical properties of `sample`.
    let rng = MersenneTwister(123456)
        x, x′ = randn(rng, 10), randn(rng, 11)
        f = GP(sin, EQ(), GPC())

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        S = 100000
        f̂ = sample(rng, f(x), S)

        μ̂, μ = mean(f̂, 2), sin.(x)
        @test mean(abs.(μ̂ .- μ)) < 1e-2

        Σ̂, Σ = (f̂ .- sin.(x)) * (f̂ .- sin.(x)).' ./ S, cov(kernel(f(x)))
        @test mean(abs.(Σ̂ .- Σ)) < 1e-2
    
        # Check mean + covariance estimates approximately converge for multi-GP sampling.
        S, x̂ = 100000, vcat(x, x′)
        f̂ = sample(rng, Vector{GP}([f(x), f(x′)]), S)
        f̂ = vcat(f̂[1], f̂[2])

        μ̂, μ = mean(f̂, 2), sin.(x̂)
        @test mean(abs.(μ̂ .- μ)) < 1e-2

        Σ̂, Σ = (f̂ .- sin.(x̂)) * (f̂ .- sin.(x̂)).' ./ S, cov(kernel(f(x̂)))
        @test mean(abs.(Σ̂ .- Σ)) < 1e-2
    end
end
