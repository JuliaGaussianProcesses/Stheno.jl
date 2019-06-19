using LinearAlgebra

"""
    check_consistency(rng::AbstractRNG, θ, f, x::AV, y::AV, A, _to_psd, z::AV)

Some basic consistency checks for the function `f(θ)::Tuple{GP, GP}`. Mainly just checks
that Zygote works properly for `f`, and correctly derives the gradients w.r.t. `θ` for
`rand`, `logpdf`, `elbo` when considering the f.d.d.s `f(x, Σ)` and observations `y`, where
`Σ = _to_psd(A)`. The first output of `f` will be the GP sampled from and whose `logpdf`
will be computed, while the second will be used as the process for the pseudo-points, whose
inputs are `z`.
"""
function check_consistency(rng::AbstractRNG, θ, f, x::AV, y::AV, A, _to_psd, z::AV, B)

    # Check input consistency to prevent test failures for the wrong reasons.
    @assert length(x) == length(y)
    @assert size(Σ, 1) == length(y)
    @assert size(Σ, 1) == size(Σ, 2)
    @assert all(eigvals(Σ) > 0)
    @assert issymmetric(Σ)

    g = (θ, x, A)->FiniteGP(first(f(θ)), x, _to_psd(A))
    function h(θ, x, A, z, B)
        f, u = f(θ)
        return FiniteGP(f, x, _to_psd(A)), FiniteGP(u, z, _to_psd(B))
    end

    # Check that the gradient w.r.t. the samples is correct (single-sample).
    adjoint_test(
        (θ, x, A)->rand(MersenneTwister(123456), g(θ, x, A)),
        randn(rng, length(x)),
        θ, x, A,
    )

    # Check that the gradient w.r.t. the samples is correct (multi-sample).
    adjoint_test(
        (θ, x, A)->rand(MersenneTwister(123456), g(θ, x, A), 11),
        randn(rng, length(x), 11),
        θ, x, A,
    )

    # Check adjoints for logpdf.
    adjoint_test((θ, x, A, y)->logpdf(g(θ, x, A), y), randn(rng), θ, x, A)

    # Check adjoint for elbo.
    adjoint_test(
        (ϴ, x, A, y)->begin
            fx, uz = h(θ, x, A, z, B)
            return elbo(fx, y, uz)
        end,
        randn(rng),
        θ, x, A, y, z, B,
    )
end
