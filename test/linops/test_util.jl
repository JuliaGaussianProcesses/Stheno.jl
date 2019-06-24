using LinearAlgebra
using Stheno: FiniteGP, BlockKernel

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

    # # Check that we can differentiate through evaluation of the mean vector.
    # adjoint_test(
    #     (θ, x, A)->mean(g(θ, x, A)),
    #     randn(rng, length(x)), θ, x, A;
    #     rtol=1e-4, atol=1e-4,
    # )

    # # Check that we can differentiate through evaluation of the covariance matrix.
    # adjoint_test(
    #     (θ, x, A)->cov(g(θ, x, A)),
    #     randn(rng, length(x), length(x)), θ, x, A;
    #     rtol=1e-4, atol=1e-4,
    # )

    # adjoint_test(
    #     (θ, x, A, z, B)->cov(h(θ, x, A, z, B)...),
    #     randn(rng, length(x), length(z)), θ, x, A, z, B;
    #     rtol=1e-4, atol=1e-4,
    # )

    # If the above two tests pass, then the ones below should also pass. Test them anyway.

    # # Check that the gradient w.r.t. the samples is correct (single-sample).
    # adjoint_test(
    #     (θ, x, A)->rand(MersenneTwister(123456), g(θ, x, A)),
    #     randn(rng, length(x)), θ, x, A;
    #     rtol=1e-4, atol=1e-4,
    # )

    # # Check that the gradient w.r.t. the samples is correct (multi-sample).
    # adjoint_test(
    #     (θ, x, A)->rand(MersenneTwister(123456), g(θ, x, A), 11),
    #     randn(rng, length(x), 11), θ, x, A;
    #     rtol=1e-4, atol=1e-4,
    # )

    # # Check adjoints for logpdf.
    # adjoint_test(
    #     (θ, x, A, y)->logpdf(g(θ, x, A), y), randn(rng), θ, x, A, y;
    #     rtol=1e-4, atol=1e-4,
    # )

    # Check adjoint for elbo.
    adjoint_test(
        (ϴ, x, A, y, z, B)->begin
            fx, uz = h(θ, x, A, z, B)
            return elbo(fx, y, uz)
        end,
        randn(rng), θ, x, A, y, z, B;
        rtol=1e-4, atol=1e-4,
    )
end

inputs(k::Kernel, N::Int) = collect(range(-5.0, 5.0; length=N))
function inputs(k::BlockKernel, N::Int)
    if size(k.ks, 1) == 1
        return BlockData(collect(range(-5.0, 5.0; length=N)))
    elseif size(k.ks, 1) == 2
        x = collect(range(-5.0, 5.0; length=N))
        return BlockData([x[1:div(N, 2)], x[div(N, 2) + 1:end]])
    else
        error("Incorrectly-sized block kernel")
    end
end

function standard_1D_dense_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    g, u = f(θ)
    x, z = inputs(kernel(g), N), inputs(kernel(u), M)
    A, B = randn(rng, N, N), randn(rng, M, M)
    y = rand(rng, first(f(θ))(x, _to_psd(A)))
    check_consistency(rng, θ, f, x, y, A, z, B)
end

function standard_1D_diag_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    g, u = f(θ)
    x, z = inputs(kernel(g), N), inputs(kernel(u), M)
    a, b = randn(rng, N), randn(rng, M)
    y = rand(rng, first(f(θ))(x, _to_psd(a)))
    check_consistency(rng, θ, f, x, y, a, z, b)
end

function standard_1D_isotropic_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    g, u = f(θ)
    x, z = inputs(kernel(g), N), inputs(kernel(u), M)
    a, b = randn(rng), randn(rng)
    y = rand(rng, first(f(θ))(x, _to_psd(a)))
    check_consistency(rng, θ, f, x, y, a, z, b)
end

function standard_1D_block_diagonal_test(rng::AbstractRNG, θ, f, N::Int, M::Int)
    @assert N > 2 && M > 3
    g, u = f(θ)
    x, z = inputs(kernel(g), N), inputs(kernel(u), M)
    As = [randn(rng, 2, 2), randn(rng, N - 2, N - 2)]
    Bs = [randn(rng, 3, 3), randn(rng, M - 3, M - 3)]
    y = rand(rng, first(f(θ))(x, _to_psd(As)))
    check_consistency(rng, θ, f, x, y, As, z, Bs)
end

function standard_1D_tests(rng::AbstractRNG, θ, f, N::Int, M::Int)
    standard_1D_dense_test(rng, θ, f, N, M)
    standard_1D_diag_test(rng, θ, f, N, M)
    standard_1D_isotropic_test(rng, θ, f, N, M)
    standard_1D_block_diagonal_test(rng, θ, f, N, M)
end
