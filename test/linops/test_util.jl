using LinearAlgebra
using Stheno: FiniteGP

"""
    check_consistency(rng::AbstractRNG, θ, f, x::AV, y::AV, A, z::AV)

Some basic consistency checks for the function `f(θ)::Tuple{GP, GP}`. Mainly just checks
that Zygote works properly for `f`, and correctly derives the gradients w.r.t. `θ` for
`rand`, `logpdf`, `elbo` when considering the f.d.d.s `f(x, Σ)` and observations `y`, where
`Σ = _to_psd(A)`. The first output of `f` will be the GP sampled from and whose `logpdf`
will be computed, while the second will be used as the process for the pseudo-points, whose
inputs are `z`.
"""
function check_consistency(rng::AbstractRNG, θ, f, x::AV, y::AV, A, z::AV, B)

    # Check input consistency to prevent test failures for the wrong reasons.
    @assert length(x) == length(y)

    g = (θ, x, A)->FiniteGP(first(f(θ)), x, _to_psd(A))
    function h(θ, x, A, z, B)
        g, u = f(θ)
        return FiniteGP(g, x, _to_psd(A)), FiniteGP(u, z, _to_psd(B))
    end

    # Check that the gradient w.r.t. the samples is correct (single-sample).
    adjoint_test(
        (θ, x, A)->rand(MersenneTwister(123456), g(θ, x, A)),
        randn(rng, length(x)),
        θ, x, A;
        rtol=1e-7, atol=1e-7,
    )

    # Check that the gradient w.r.t. the samples is correct (multi-sample).
    adjoint_test(
        (θ, x, A)->rand(MersenneTwister(123456), g(θ, x, A), 11),
        randn(rng, length(x), 11),
        θ, x, A;
        rtol=1e-7, atol=1e-7,
    )

    # Check adjoints for logpdf.
    adjoint_test(
        (θ, x, A, y)->logpdf(g(θ, x, A), y), randn(rng), θ, x, A, y;
        rtol=1e-7, atol=1e-7,
    )

    # Check adjoint for elbo.
    adjoint_test(
        (ϴ, x, A, y, z, B)->begin
            fx, uz = h(θ, x, A, z, B)
            return elbo(fx, y, uz)
        end,
        randn(rng),
        θ, x, A, y, z, B;
        rtol=1e-7, atol=1e-7,
    )
end

function standard_1D_dense_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    x, z = collect(range(-5.0, 5.0; length=N)), collect(range(-5.0, 5.0; length=M))
    A, B = randn(rng, N, N), randn(rng, M, M)
    y = rand(rng, first(f(θ))(x, _to_psd(A)))
    check_consistency(rng, θ, f, x, y, A, z, B)
end

function standard_1D_diag_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    x, z = collect(range(-5.0, 5.0; length=N)), collect(range(-5.0, 5.0; length=M))
    a, b = randn(rng, N), randn(rng, M)
    y = rand(rng, first(f(θ))(x, _to_psd(a)))
    check_consistency(rng, θ, f, x, y, a, z, b)
end

function standard_1D_isotropic_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    x, z = collect(range(-5.0, 5.0; length=N)), collect(range(-5.0, 5.0; length=M))
    a, b = randn(rng), randn(rng)
    y = rand(rng, first(f(θ))(x, _to_psd(a)))
    check_consistency(rng, θ, f, x, y, a, z, b)
end

function standard_1D_tests(rng::AbstractRNG, θ, f, N::Int, M::Int)
    standard_1D_dense_test(rng, θ, f, N, M)
    standard_1D_diag_test(rng, θ, f, N, M)
    standard_1D_isotropic_test(rng, θ, f, N, M)
end
