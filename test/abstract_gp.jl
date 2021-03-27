using Stheno: FiniteGP, ConstMean, block_diagonal, tr_Cf_invΣy
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

        @test mean(fx) == mean_vector(f, x)
        @test cov(fx) == cov(f, x)
        @test cov(fx, fx′) == cov(f, x, x′)
        @test mean.(marginals(fx)) == mean(f(x))
        @test var.(marginals(fx)) == cov_diag(f, x)
        @test std.(marginals(fx)) == sqrt.(cov_diag(f, x))
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
    # @testset "tr_Cf_invΣy" begin
    #     N = 11
    #     x = collect(range(-3.0, 3.0; length=N))
    #     @testset "dense" begin
    #         rng = MersenneTwister(123456)
    #         A = randn(rng, N, N - 2)
    #         adjoint_test(
    #             (x, A)->begin
    #                 f = GP(sin, SqExponentialKernel(), GPC())
    #                 Σy = _to_psd(A)
    #                 C = cholesky(Σy)
    #                 return tr_Cf_invΣy(FiniteGP(f, x, Σy), Σy, C)
    #             end,
    #             randn(rng), x, A,
    #         )
    #     end
    #     @testset "Diagonal" begin
    #         rng = MersenneTwister(123456)
    #         a = 0.01 .* randn(rng, N)
    #         adjoint_test(
    #             (x, a)->begin
    #                 f = GP(sin, SqExponentialKernel(), GPC())
    #                 Σy = Diagonal(exp.(a .+ 1))
    #                 C = cholesky(Σy)
    #                 return tr_Cf_invΣy(FiniteGP(f, x, Σy), Σy, C)
    #             end,
    #             randn(rng), x, a,
    #         )
    #     end
    #     # @testset "BlockDiagonal" begin
    #     #     rng = MersenneTwister(123456)
    #     #     A1, A2 = randn(rng, N, 4), randn(rng, N+1, 5)
    #     #     x = collect(range(-5.0, 5.0; length=size(A1, 1) + size(A2, 1)))
    #     #     Nx = length(x)
    #     #     adjoint_test(
    #     #         (x, A1, A2)->begin
    #     #             f = GP(sin, SqExponentialKernel(), GPC())
    #     #             Σ1, Σ2 = _to_psd(A1), _to_psd(A2)
    #     #             Σy = block_diagonal([Σ1, Σ2])
    #     #             C = cholesky(Σy)
    #     #             return tr_Cf_invΣy(FiniteGP(f, x, Σy), Σy, C)
    #     #         end,
    #     #         randn(rng), x, A1, A2;
    #     #         atol=1e-6, rtol=1e-6,
    #     #     )
    #     # end
    # end
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

# """
#     simple_gp_tests(rng::AbstractRNG, f::GP, xs::AV{<:AV}, σs::AV{<:Real})

# Integration tests for simple GPs.
# """
# function simple_gp_tests(
#     rng::AbstractRNG,
#     f::GP,
#     xs::AV{<:AV},
#     isp_σs::AV{<:Real};
#     atol=1e-8,
#     rtol=1e-8,
# )
#     for x in xs, isp_σ in isp_σs

#         # Test gradient w.r.t. random sampling.
#         N = length(x)
#         adjoint_test(
#             (x, isp_σ)->rand(_rng(), FiniteGP(f, x, exp(isp_σ)^2)),
#             randn(rng, N),
#             x,
#             isp_σ,;
#             atol=atol, rtol=rtol,
#         )
#         adjoint_test(
#             (x, isp_σ)->rand(_rng(), FiniteGP(f, x, exp(isp_σ)^2), 11),
#             randn(rng, N, 11),
#             x,
#             isp_σ,;
#             atol=atol, rtol=rtol,
#         )

#         # Check that gradient w.r.t. logpdf is correct.
#         y, l̄ = rand(rng, FiniteGP(f, x, exp(isp_σ))), randn(rng)
#         adjoint_test(
#             (x, isp_σ, y)->logpdf(FiniteGP(f, x, exp(isp_σ)), y),
#             l̄, x, isp_σ, y;
#             atol=atol, rtol=rtol,
#         )

#         # Check that elbo is tight-ish when it's meant to be.
#         fx, yx = FiniteGP(f, x, 1e-9), FiniteGP(f, x, exp(isp_σ))
#         @test isapprox(elbo(yx, y, fx), logpdf(yx, y); atol=1e-6, rtol=1e-6)

#         # Check that gradient w.r.t. elbo is correct.
#         adjoint_test(
#             (x, ŷ, isp_σ)->elbo(FiniteGP(f, x, exp(isp_σ)), ŷ, FiniteGP(f, x, 1e-9)),
#             randn(rng), x, y, isp_σ;
#             atol=1e-6, rtol=1e-6,
#         )
#     end
# end

# __foo(x) = isnothing(x) ? "nothing" : x

# @testset "FiniteGP (integration)" begin
#     rng = MersenneTwister(123456)
#     xs = [collect(range(-3.0, stop=3.0, length=N)) for N in [2, 5, 10]]
#     σs = log.([1e-1, 1e0, 1e1])
#     for (k, name, atol, rtol) in vcat(
#         [
#             (SqExponentialKernel(), "SqExponentialKernel", 1e-6, 1e-6),
#             (LinearKernel(), "Linear", 1e-6, 1e-6),
#             (PerSqExponentialKernel(), "PerSqExponentialKernel", 5e-5, 1e-8),
#             (ExponentialKernel(), "Exp", 1e-6, 1e-6),
#         ],
#         [(
#             k(α=α, β=β, l=l), 
#             "$k_name(α=$(__foo(α)), β=$(__foo(β)), l=$(__foo(l)))",
#             1e-6,
#             1e-6,
#         )
#             for (k, k_name) in ((SqExponentialKernel, "SqExponentialKernel"), (Linear, "linear"), (Matern12, "exp"))
#             for α in (nothing, randn(rng))
#             for β in (nothing, exp(randn(rng)))
#             for l in (nothing, randn(rng))
#         ],
#     )
#         @testset "$name" begin
#             simple_gp_tests(_rng(), GP(k, GPC()), xs, σs; atol=atol, rtol=rtol)
#         end
#     end
# end

# @testset "FiniteGP (BlockDiagonal obs noise)" begin
#     rng, Ns = MersenneTwister(123456), [4, 5]
#     x = collect(range(-5.0, 5.0; length=sum(Ns)))
#     As = [randn(rng, N, N) for N in Ns]
#     Ss = [A' * A + I for A in As]

#     S = block_diagonal(Ss)
#     Smat = Matrix(S)

#     f = GP(cos, SqExponentialKernel(), GPC())
#     y = rand(FiniteGP(f, x, S))

#     @test logpdf(FiniteGP(f, x, S), y) ≈ logpdf(FiniteGP(f, x, Smat), y)
#     adjoint_test(
#         (x, S, y)->logpdf(FiniteGP(f, x, S), y), randn(rng), x, Smat, y;
#         atol=1e-6, rtol=1e-6,
#     )
#     adjoint_test(
#         (x, A1, A2, y)->logpdf(FiniteGP(f, x, block_diagonal([A1 * A1' + I, A2 * A2' + I])), y),
#         randn(rng), x, As[1], As[2], y;
#         atol=1e-6, rtol=1e-6
#     )

#     @test elbo(FiniteGP(f, x, Smat), y, FiniteGP(f, x)) ≈ logpdf(FiniteGP(f, x, Smat), y)
#     @test elbo(FiniteGP(f, x, S), y, FiniteGP(f, x)) ≈
#         elbo(FiniteGP(f, x, Smat), y, FiniteGP(f, x))
#     adjoint_test(
#         (x, A, y)->elbo(FiniteGP(f, x, _to_psd(A)), y, FiniteGP(f, x)),
#         randn(rng), x, randn(rng, sum(Ns), sum(Ns)), y;
#         atol=1e-6, rtol=1e-6,
#     )
#     adjoint_test(
#         (x, A1, A2, y) -> begin
#             S = block_diagonal([A1 * A1' + I, A2 * A2' + I])
#             return elbo(FiniteGP(f, x, S), y, FiniteGP(f, x))
#         end,
#         randn(rng), x, As[1], As[2], y;
#         atol=1e-6, rtol=1e-6,
#     )
# end
