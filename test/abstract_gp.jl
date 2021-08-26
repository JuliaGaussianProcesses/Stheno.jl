_rng() = MersenneTwister(123456)

function generate_noise_matrix(rng::AbstractRNG, N::Int)
    A = randn(rng, N, N)
    return Symmetric(A * A' + I)
end

@testset "finite_gp" begin
    @testset "statistics" begin
        rng, N, N′ = MersenneTwister(123456), 1, 9
        x, x′, Σy, Σy′ = randn(rng, N), randn(rng, N′), zeros(N, N), zeros(N′, N′)
        f = wrap(GP(sin, SEKernel()), GPC())
        fx, fx′ = FiniteGP(f, x, Σy), FiniteGP(f, x′, Σy′)

        @test mean(fx) == mean(f, x)
        @test cov(fx) == cov(f, x)
        @test cov(fx, fx′) == cov(f, x, x′)
        @test mean.(marginals(fx)) == mean(f(x))
        @test var.(marginals(fx)) == var(f, x)
        @test std.(marginals(fx)) == sqrt.(var(f, x))
    end
    @testset "rand (deterministic)" begin
        rng, N, D = MersenneTwister(123456), 10, 2
        X, x, Σy = ColVecs(randn(rng, D, N)), randn(rng, N), zeros(N, N)
        Σy = generate_noise_matrix(rng, N)
        fX = FiniteGP(wrap(GP(1, SEKernel()), GPC()), X, Σy)
        fx = FiniteGP(wrap(GP(1, SEKernel()), GPC()), x, Σy)

        # Check that single-GP samples have the correct dimensions.
        @test length(rand(rng, fX)) == length(X)
        @test size(rand(rng, fX, 10)) == (length(X), 10)

        @test length(rand(rng, fx)) == length(x)
        @test size(rand(rng, fx, 10)) == (length(x), 10)
    end
    @testset "rand (statistical)" begin
        rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
        X, Σy = ColVecs(randn(rng, D, N)), 1e-12
        f = FiniteGP(wrap(GP(1, SEKernel()), GPC()), X, Σy)

        # Check mean + covariance estimates approximately converge for single-GP sampling.
        f̂ = rand(rng, f, S)
        @test maximum(abs.(mean(f̂; dims=2) - mean(f))) < 1e-2

        Σ′ = (f̂ .- mean(f)) * (f̂ .- mean(f))' ./ S
        @test mean(abs.(Σ′ - cov(f))) < 1e-2
    end
    @testset "rand (gradients)" begin
        rng, N, S = MersenneTwister(123456), 10, 3
        x = collect(range(-3.0, stop=3.0, length=N))
        Σy = 1e-12

        # Check that the gradient w.r.t. the samples is correct (single-sample).
        adjoint_test(
            x->rand(
                MersenneTwister(123456),
                FiniteGP(wrap(GP(sin, SEKernel()), GPC()), x, Σy),
            ),
            randn(rng, N),
            x;
            atol=1e-9, rtol=1e-9,
        )

        # Check that the gradient w.r.t. the samples is correct (multisample).
        adjoint_test(
            x->rand(
                MersenneTwister(123456),
                FiniteGP(wrap(GP(sin, SEKernel()), GPC()), x, Σy),
                S,
            ),
            randn(rng, N, S),
            x;
            atol=1e-9, rtol=1e-9,
        )
    end
    @testset "Type Stability - $T" for T in [Float64, Float32]
        rng = MersenneTwister(123456)
        x = randn(rng, T, 123)
        z = randn(rng, T, 13)
        f = wrap(GP(T(0), SEKernel()), GPC())

        fx = f(x, T(0.1))
        u = f(z, T(1e-4))

        y = rand(rng, fx)
        @test y isa Vector{T}
        @test logpdf(fx, y) isa T
    end
end
