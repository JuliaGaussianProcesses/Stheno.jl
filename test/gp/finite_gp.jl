using Stheno: FiniteGP, GPC, pw, ConstMean, OuterKernel, AbstractGP, BlockGP
using Stheno: EQ, Exp, Linear, Noise, PerEQ
using Statistics, StatsFuns
using Distributions: MvNormal, PDMat

_rng() = MersenneTwister(123456)

function generate_noise_matrix(rng::AbstractRNG, N::Int)
    A = randn(rng, N, N)
    return Symmetric(A * A' + I)
end

@testset "finite_gp" begin

    # @testset "statistics" begin
    #     rng, N, N′ = MersenneTwister(123456), 1, 9
    #     x, x′, Σy, Σy′ = randn(rng, N), randn(rng, N′), Zeros(N, N), Zeros(N′, N′)
    #     f = GP(sin, EQ(), GPC())
    #     fx, fx′ = FiniteGP(f, x, Σy), FiniteGP(f, x′, Σy′)

    #     @test mean(fx) == map(mean(f), x)
    #     @test cov(fx) == pw(kernel(f), x)
    #     @test cov(fx, fx′) == pw(kernel(f), x, x′)
    #     @test mean.(marginals(fx)) == mean(f(x))
    #     @test var.(marginals(fx)) == map(kernel(f), x)
    #     @test std.(marginals(fx)) == sqrt.(map(kernel(f), x))
    # end

    # @testset "rand (deterministic)" begin
    #     rng, N, D = MersenneTwister(123456), 10, 2
    #     X, x, Σy = ColsAreObs(randn(rng, D, N)), randn(rng, N), Zeros(N, N)
    #     fX = FiniteGP(GP(1, EQ(), GPC()), X, Σy)
    #     fx = FiniteGP(GP(1, EQ(), GPC()), x, Σy)

    #     # Check that single-GP samples have the correct dimensions.
    #     @test length(rand(rng, fX)) == length(X)
    #     @test size(rand(rng, fX, 10)) == (length(X), 10)

    #     @test length(rand(rng, fx)) == length(x)
    #     @test size(rand(rng, fx, 10)) == (length(x), 10)
    # end

    # @testset "rand (statistical)" begin
    #     rng, N, D, μ0, S = MersenneTwister(123456), 10, 2, 1, 100_000
    #     X, Σy = ColsAreObs(randn(rng, D, N)), Zeros(N, N)
    #     f = FiniteGP(GP(1, EQ(), GPC()), X, Σy)

    #     # Check mean + covariance estimates approximately converge for single-GP sampling.
    #     f̂ = rand(rng, f, S)
    #     @test maximum(abs.(mean(f̂; dims=2) - mean(f))) < 1e-2

    #     Σ′ = (f̂ .- mean(f)) * (f̂ .- mean(f))' ./ S
    #     @test mean(abs.(Σ′ - cov(f))) < 1e-2
    # end

    # @testset "rand (gradients)" begin
    #     rng, N, S = MersenneTwister(123456), 10, 3
    #     x = collect(range(-3.0, stop=3.0, length=N))
    #     Σy = Zeros(N, N)

    #     # Check that the gradient w.r.t. the samples is correct (single-sample).
    #     adjoint_test(
    #         x->rand(MersenneTwister(123456), FiniteGP(GP(sin, EQ(), GPC()), x, Σy)),
    #         randn(rng, N),
    #         x,
    #     )

    #     # Check that the gradient w.r.t. the samples is correct (multisample).
    #     adjoint_test(
    #         x->rand(MersenneTwister(123456), FiniteGP(GP(sin, EQ(), GPC()), x, Σy), S),
    #         randn(rng, N, S),
    #         x,
    #     )
    # end

    # @testset "rand (block - deteministic)" begin
    #     rng, N, N′, S = MersenneTwister(123456), 11, 3, 7
    #     x, x′ = randn(rng, N), randn(rng, N′)
    #     f = GP(sin, eq(), GPC())
    #     fx, fx′ = FiniteGP(f, x, 1e-3), FiniteGP(f, x′, 1e-3)
    #     f_blk = BlockGP([f, f])
    #     f_blk_xx′ = FiniteGP(f_blk, BlockData([x, x′]), 1e-3)

    #     @test length(rand(rng, f_blk_xx′)) == N + N′
    #     @test size(rand(rng, f_blk_xx′, S)) == (N + N′, S)

    #     @test length(rand(rng, [fx, fx′])[1]) == N
    #     @test length(rand(rng, [fx, fx′])[2]) == N′
    #     @test size(rand(rng, [fx ,fx′], S)[1]) == (N, S)
    #     @test size(rand(rng, [fx, fx′], S)[2]) == (N′, S)

    #     Y = rand(MersenneTwister(123456), f_blk_xx′, S)
    #     Ŷ = vcat(rand(MersenneTwister(123456), [fx, fx′], S)...)
    #     @test Y == Ŷ
    # end

    # @testset "rand (block - statistical)" begin
    #     rng, N, N′, S = MersenneTwister(123456), 11, 3, 100_000
    #     x, x′ = randn(rng, N), randn(rng, N′)
    #     f = GP(cos, eq(), GPC())
    #     f_blk_xx′ = FiniteGP(BlockGP([f, f]), BlockData([x, x′]), 1e-3)

    #     f̂ = rand(rng, f_blk_xx′, S)
    #     @test maximum(abs.(mean(f̂; dims=2) - mean(f_blk_xx′))) < 1e-2

    #     Σ′ = (f̂ .- mean(f_blk_xx′)) * (f̂ .- mean(f_blk_xx′))' ./ S
    #     @test mean(abs.(Σ′ - cov(f_blk_xx′))) < 1e-2
    # end

    # @testset "rand (block - gradients)" begin
    #     rng, N, N′, S = MersenneTwister(123456), 11, 3, 10
    #     f = GP(cos, eq(), GPC())
    #     xx′ = collect(range(-3.0, stop=3.0, length=N + N′))
    #     x, x′ = xx′[1:N], xx′[N+1:end]

    #     foo(x, x′) = begin
    #         f = GP(sin, eq(), GPC())
    #         return FiniteGP(BlockGP([f, f]), BlockData([x, x′]), eps())
    #     end

    #     # Check that the gradient w.r.t. the samples is correct (single-sample).
    #     adjoint_test(
    #         (x, x′)->rand(_rng(), foo(x, x′)), randn(rng, N + N′), x, x′;
    #         rtol=1e-6,atol=1e-6,
    #     )

    #     # Check that the gradient w.r.t. the samples is correct (multisample).
    #     adjoint_test(
    #         (x, x′)->rand(_rng(), foo(x, x′), S), randn(rng, N + N′, S), x, x′;
    #         rtol=1e-6, atol=1e-6,
    #     )
    # end

    @testset "logpdf / elbo" begin
        rng, N, σ, gpc = MersenneTwister(123456), 10, 1e-1, GPC()
        x = collect(range(-3.0, stop=3.0, length=N))
        f = GP(1, eq(), gpc)
        fx, y = FiniteGP(f, x, 0), FiniteGP(f, x, σ^2)
        ŷ = rand(rng, y)

        # Check that logpdf returns the correct type and roughly agrees with Distributions.
        # @test logpdf(y, ŷ) isa Real
        # @test logpdf(y, ŷ) ≈ logpdf(MvNormal(Vector(mean(y)), cov(y)), ŷ)

        # Check gradient of logpdf at mean is zero for `f`.
        # adjoint_test(ŷ->logpdf(fx, ŷ), 1, ones(size(ŷ)))
        lp, back = Zygote.forward(ŷ->logpdf(fx, ŷ), ones(size(ŷ)))
        # @test back(randn(rng))[1] == zeros(size(ŷ)) 

        # Check that gradient of logpdf at mean is zero for `y`.
        # adjoint_test(ŷ->logpdf(y, ŷ), 1, ones(size(ŷ)))
        lp, back = Zygote.forward(ŷ->logpdf(y, ŷ), ones(size(ŷ)))
        # @test back(randn(rng))[1] == zeros(size(ŷ))

        # Check that gradient w.r.t. inputs is approximately correct for `f`.
        x, l̄ = randn(rng, N), randn(rng)
        # adjoint_test(x->logpdf(FiniteGP(f, x, 1e-3), ones(size(x))), l̄, collect(x))

        # Check that the gradient w.r.t. the noise is approximately correct for `f`.
        # adjoint_test(σ_->logpdf(FiniteGP(f, x, softplus(σ_)), ŷ), l̄, randn(rng))

        # Check that the gradient w.r.t. a scaling of the GP works.
        adjoint_test(α->logpdf(FiniteGP(α * f, x, 1e-1), ŷ), l̄, randn(rng))

        # # Ensure that the elbo is close to the logpdf when appropriate.
        # @test elbo(y, ŷ, fx) isa Real
        # @test elbo(y, ŷ, fx) ≈ logpdf(y, ŷ)
        # @test elbo(y, ŷ, y) < logpdf(y, ŷ)
        # @test elbo(y, ŷ, FiniteGP(f, x, 2 * σ^2)) < elbo(y, ŷ, y)

        # # Check adjoint w.r.t. elbo actually works with the new syntax.
        # adjoint_test(
        #     (x, ŷ, σ)->elbo(FiniteGP(f, x, σ^2), ŷ, FiniteGP(f, x, 0)),
        #     randn(rng), x, ŷ, σ;
        #     atol=1e-6, rtol=1e-6,
        # )
    end
end

"""
    simple_gp_tests(rng::AbstractRNG, f::AbstractGP, xs::AV{<:AV}, σs::AV{<:Real})

Integration tests for simple GPs.
"""
function simple_gp_tests(
    rng::AbstractRNG,
    f::AbstractGP,
    xs::AV{<:AV},
    isp_σs::AV{<:Real};
    atol=1e-8,
    rtol=1e-8,
)
    for x in xs, isp_σ in isp_σs

        # Test gradient w.r.t. random sampling.
        N = length(x)
        adjoint_test(
            (x, isp_σ)->rand(_rng(), FiniteGP(f, x, softplus(isp_σ)^2)),
            randn(rng, N),
            x,
            isp_σ,;
            atol=atol, rtol=rtol,
        )    
        adjoint_test(
            (x, isp_σ)->rand(_rng(), FiniteGP(f, x, softplus(isp_σ)^2), 11),
            randn(rng, N, 11),
            x,
            isp_σ,;
            atol=atol, rtol=rtol,
        )

        # Check that gradient w.r.t. logpdf is correct.
        y, l̄ = rand(rng, FiniteGP(f, x, softplus(isp_σ))), randn(rng)
        adjoint_test(
            (x, isp_σ, y)->logpdf(FiniteGP(f, x, softplus(isp_σ)), y),
            l̄, x, isp_σ, y;
            atol=atol, rtol=rtol,
        )

        # Check that elbo is tight-ish when it's meant to be.
        fx, yx = FiniteGP(f, x, 1e-9), FiniteGP(f, x, softplus(isp_σ))
        @test isapprox(elbo(yx, y, fx), logpdf(yx, y); atol=1e-6, rtol=1e-6)

        # Check that gradient w.r.t. elbo is correct.
        adjoint_test(
            (x, ŷ, isp_σ)->elbo(FiniteGP(f, x, softplus(isp_σ)), ŷ, FiniteGP(f, x, 1e-9)),
            randn(rng), x, y, isp_σ;
            atol=1e-6, rtol=1e-6,
        )
    end
end

__foo(x) = isnothing(x) ? "nothing" : x

# @testset "FiniteGP (integration)" begin
#     rng = MersenneTwister(123456)
#     xs = [collect(range(-3.0, stop=3.0, length=N)) for N in [2, 5, 10]]
#     # xs = [collect(range(-3.0, stop=3.0, length=10))]
#     σs = invsoftplus.([1e-1, 1e0, 1e1])
#     for (k, name, atol, rtol) in vcat(
#         [
#             (EQ(), "EQ", 1e-8, 1e-8),
#             (Linear(), "Linear", 1e-8, 1e-8),
#             (PerEQ(), "PerEQ", 5e-5, 1e-8),
#             (Exp(), "Exp", 1e-8, 1e-8),
#         ],
#         [(
#             k(α=α, β=β, l=l), 
#             "$k_name(α=$(__foo(α)), β=$(__foo(β)), l=$(__foo(l)))",
#             1e-8,
#             1e-8,
#         )
#             for (k, k_name) in ((eq, "eq"), (linear, "linear"), (exp, "exp"))
#             for α in (nothing, randn(rng))
#             for β in (nothing, softplus(randn(rng)))
#             for l in (nothing, randn(rng))
#         ],
#     )
#         @testset "$name" begin
#             simple_gp_tests(_rng(), GP(k, GPC()), xs, σs; atol=atol, rtol=rtol)
#         end
#     end
# end
