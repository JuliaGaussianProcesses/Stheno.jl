using Stheno: FiniteGP, block_diagonal
using Distributions: MvNormal

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
    @testset "logpdf / elbo / dtc" begin
        rng, N, S, σ, gpc = MersenneTwister(123456), 10, 11, 1e-1, GPC()
        x = collect(range(-3.0, stop=3.0, length=N))
        f = wrap(GP(1, SEKernel()), gpc)
        fx, y = FiniteGP(f, x, 0), FiniteGP(f, x, σ^2)
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        @test logpdf(y, ŷ) isa Real
        @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(mean(y)), cov(y)), ŷ)

        # Check that multi-sample logpdf returns the correct type and is consistent with
        # single-sample logpdf
        Ŷ = rand(rng, y, S)
        @test logpdf(y, Ŷ) isa Vector{Float64}
        @test logpdf(y, Ŷ) ≈ [logpdf(y, Ŷ[:, n]) for n in 1:S]

        # Check gradient of logpdf at mean is zero for `f`.
        adjoint_test(ŷ->logpdf(fx, ŷ), 1, ones(size(ŷ)))
        lp, back = Zygote.pullback(ŷ->logpdf(fx, ŷ), ones(size(ŷ)))
        @test back(randn(rng))[1] == zeros(size(ŷ))

        # Check that gradient of logpdf at mean is zero for `y`.
        adjoint_test(ŷ->logpdf(y, ŷ), 1, ones(size(ŷ)))
        lp, back = Zygote.pullback(ŷ->logpdf(y, ŷ), ones(size(ŷ)))
        @test back(randn(rng))[1] == zeros(size(ŷ))

        # Check that gradient w.r.t. inputs is approximately correct for `f`.
        x, l̄ = randn(rng, N), randn(rng)
        adjoint_test(
            x->logpdf(FiniteGP(f, x, 1e-3), ones(size(x))),
            l̄, collect(x);
            atol=1e-8, rtol=1e-8,
        )
        adjoint_test(
            x->sum(logpdf(FiniteGP(f, x, 1e-3), ones(size(Ŷ)))),
            l̄, collect(x);
            atol=1e-8, rtol=1e-8,
        )

        # Check that the gradient w.r.t. the noise is approximately correct for `f`.
        σ_ = randn(rng)
        adjoint_test((σ_, ŷ)->logpdf(FiniteGP(f, x, exp(σ_)), ŷ), l̄, σ_, ŷ)
        adjoint_test((σ_, Ŷ)->sum(logpdf(FiniteGP(f, x, exp(σ_)), Ŷ)), l̄, σ_, Ŷ)

        # Check that the gradient w.r.t. a scaling of the GP works.
        adjoint_test(
            α->logpdf(FiniteGP(α * f, x, 1e-1), ŷ), l̄, randn(rng);
            atol=1e-8, rtol=1e-8,
        )
        adjoint_test(
            α->sum(logpdf(FiniteGP(α * f, x, 1e-1), Ŷ)), l̄, randn(rng);
            atol=1e-8, rtol=1e-8,
        )

        # Ensure that the elbo is close to the logpdf when appropriate.
        @test elbo(y, ŷ, fx) isa Real
        @test elbo(y, ŷ, fx) ≈ logpdf(y, ŷ)
        @test elbo(y, ŷ, y) < logpdf(y, ŷ)
        @test elbo(y, ŷ, FiniteGP(f, x, 2 * σ^2)) < elbo(y, ŷ, y)

        # Check adjoint w.r.t. elbo is correct.
        adjoint_test(
            (x, ŷ, σ)->elbo(FiniteGP(f, x, σ^2), ŷ, FiniteGP(f, x, 0)),
            randn(rng), x, ŷ, σ;
            atol=1e-6, rtol=1e-6,
        )

        # Ensure that the dtc is close to the logpdf when appropriate.
        @test dtc(y, ŷ, fx) isa Real
        @test dtc(y, ŷ, fx) ≈ logpdf(y, ŷ)

        # Check adjoint w.r.t. dtc is correct.
        adjoint_test(
            (x, ŷ, σ)->dtc(FiniteGP(f, x, σ^2), ŷ, FiniteGP(f, x, 0)),
            randn(rng), x, ŷ, σ;
            atol=1e-6, rtol=1e-6,
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
        @test elbo(fx, y, u) isa T
    end
end
