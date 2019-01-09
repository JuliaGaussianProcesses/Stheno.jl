using BlockArrays, LinearAlgebra, FDM, Zygote, ToeplitzMatrices
using Stheno: MeanFunction, Kernel, CrossKernel, AV, blocks, pairwise
using FillArrays: AbstractFill

const _rtol = 1e-10
const _atol = 1e-10

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    println()
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

# Compare FDM estimate of the adjoint with Zygotes. Also ensure that forwards passes match.
function adjoint_test(f, ȳ::AV{<:Real}, x::AV{<:Real}; rtol=_rtol, atol=_atol)

    # Compute Zygote forward-pass and ensure that it matches regular evaulation of `f`.
    y, back = Zygote.forward(f, x)
    @test y == f(x)

    # Compare gradients.
    adjoint_ad = back(ȳ)[1]
    adjoint_fd = FDM.adjoint(central_fdm(5, 1), f, ȳ, x)
    # if !(adjoint_ad isa Nothing) 
    #     print_adjoints(adjoint_ad, adjoint_fd, rtol, atol) # util for debugging.
    # end
    @test (adjoint_ad isa Nothing &&
        isapprox(adjoint_fd, zero(adjoint_fd); rtol=rtol, atol=atol)) ||
        isapprox(adjoint_ad, adjoint_fd; rtol=rtol, atol=atol)
end
function adjoint_test(f, ȳ::Real, x::AV{<:Real}; rtol=_rtol, atol=_atol)
    return adjoint_test(x->[f(x)], [ȳ], x; rtol=rtol, atol=atol)
end
function adjoint_test(f, ȳ::AV{<:Real}, x::Real; rtol=_rtol, atol=_atol)
    return adjoint_test(x->f(x[1]), ȳ, [x]; rtol=rtol, atol=atol)
end
function adjoint_test(f, ȳ::Real, x::Real; rtol=_rtol, atol=_atol)
    return adjoint_test(x->[f(x[1])], [ȳ], [x]; rtol=rtol, atol=atol)
end

# If a general array is provided, assume that `size(f(x)) == size(ȳ)`. Do appropriate
# reshaping to make stuff work in terms of vectors.
function adjoint_test(
    f,
    ȳ::AbstractArray{<:Real},
    x::AbstractArray{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    sz = size(x)
    return adjoint_test(
        x->reshape(f(reshape(x, sz...)), :),
        reshape(ȳ, :),
        reshape(x, :);
        rtol=rtol,
        atol=atol,
    )
end
function adjoint_test(f, ȳ::AbstractArray{<:Real}, x::Real; rtol=_rtol, atol=_atol)
    return adjoint_test(x->f(x[1]), ȳ, [x]; rtol=rtol, atol=atol)
end
function adjoint_test(f, ȳ::Real, x::AbstractArray{<:Real}; rtol=_rtol, atol=_atol)
    return adjoint_test(x->[f(x)], [ȳ], x; rtol=rtol, atol=atol)
end

function adjoint_test(
    f,
    ȳ::Union{Real, AbstractArray{<:Real}},
    x::ColsAreObs{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    return adjoint_test(X->f(ColsAreObs(X)), ȳ, x.X; rtol=rtol, atol=atol)
end

"""
    unary_map_tests(f, X::AbstractVector)

Consistency tests intended for use with `MeanFunction`s.
"""
function unary_map_tests(f, X::AbstractVector)
    @test map(f, X) isa AbstractVector
    @test length(f.(X)) == length(X)
    @test map(f, X) ≈ [f(x) for x in X]
end

"""
    binary_map_tests(f, X::AbstractVector, X′::AbstractVector)

Consistency tests intended for use with `CrossKernel`s.
"""
function binary_map_tests(f, X::AbstractVector, X′::AbstractVector)
    @test map(f, X, X′) isa AbstractVector
    @test length(map(f, X, X′)) == length(X)
    @test map(f, X, X′) ≈ [f(x, x′) for (x, x′) in zip(X, X′)]
end

"""
    binary_map_tests(f, X::AbstractVector)

Consistency tests intended for use with `Kernel`s.
"""
function binary_map_tests(f, X::AbstractVector)
    @test map(f, X) isa AbstractVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ map(f, X, X)
end

"""
    pairwise_tests(f, X::AbstractVector, X′::AbstractVector)

Consistency tests intended for use with `CrossKernel`s.
"""
function pairwise_tests(f, X::AbstractVector, X′::AbstractVector)
    N, N′ = length(X), length(X′)
    @test pairwise(f, X, X′) isa AbstractMatrix
    @test size(pairwise(f, X, X′)) == (N, N′)

    pw_out = pairwise(f, X, X′)
    loop_out = reshape([f(x, x′) for (x, x′) in Iterators.product(X, X′)], N, N′)
    @test all(abs.(pw_out .- loop_out) .< 1e-8)
end

"""
    pairwise_tests(f, X::AbstractVector)

Consistency tests intended for use with `Kernel`s.
"""
function pairwise_tests(f, X::AbstractVector; rtol=eps())
    @test pairwise(f, X) isa AbstractMatrix
    @test size(pairwise(f, X)) == (length(X), length(X))
    @test isapprox(pairwise(f, X), pairwise(f, X, X); atol=rtol)
end

"""
    mean_function_tests(μ::MeanFunction, X::AbstractVector)

Tests that any mean function `μ` should be able to pass.
"""
function mean_function_tests(μ::MeanFunction, X::AbstractVector)
    __mean_function_tests(μ, X)
end
function mean_function_tests(μ::MeanFunction, X::BlockData)
    __mean_function_tests(μ, X)
end
function __mean_function_tests(μ::MeanFunction, X::AbstractVector)

    # Test compulsory interface passes.
    @test !(μ(X[1]) isa Nothing)

    @test hasmethod(eachindex, Tuple{typeof(μ)})

    # Test optional interface.
    unary_map_tests(μ, X)
end

"""
    cross_kernel_tests(k::CrossKernel, X::AbstractVector, X′::AbstractVector)

Tests that any kernel `k` should be able to pass. Requires that `length(X0) == length(X1)`
and `length(X0) ≠ length(X2)`.
"""
function cross_kernel_tests(k::CrossKernel, X0::AV, X1::AV, X2::AV)
    __cross_kernel_tests(k, X0, X1, X2)
end
function cross_kernel_tests(k::CrossKernel, X0::BlockData, X1::BlockData, X2::BlockData)
    __cross_kernel_tests(k, X0, X1, X2)
end
function __cross_kernel_tests(k::CrossKernel, X0::AV, X1::AV, X2::AV)
    @assert length(X0) == length(X1)
    @assert length(X0) ≠ length(X2)

    binary_map_tests(k, X0, X1)
    pairwise_tests(k, X0, X2)
end

"""
    kernel_tests(k::Kernel, X0::AbstractVector, X1::AbstractVector, X2::AbstractVector)

Tests that any kernel `k` should be able to pass. Requires that `length(X0) == length(X1)`
and `length(X0) ≠ length(X2)`.
"""
function kernel_tests(k::Kernel, X0::AV, X1::AV, X2::AV, rtol::Real=eps())
    __kernel_tests(k, X0, X1, X2, rtol)
end
function kernel_tests(k::Kernel, X0::AV, X1::AV, X2::AV, rtol::Real=eps())
    __kernel_tests(k, X0, X1, X2, rtol)
end
function __kernel_tests(k::Kernel, X0::AV, X1::AV, X2::AV, rtol::Real=eps())
    @assert length(X0) == length(X1)
    @assert length(X0) ≠ length(X2)

    # Generic tests.
    cross_kernel_tests(k, X0, X1, X2)
    binary_map_tests(k, X0)
    pairwise_tests(k, X0; rtol=rtol)

    # Kernels should be symmetric for same arguments.
    @test pairwise(k, X0) isa AbstractMatrix
    @test pairwise(k, X0) ≈ pairwise(k, X0)'

    # k(x, x′) == k(x′, x)
    @test map(k, X0, X1) ≈ map(k, X1, X0)
    @test maximum(abs.(pairwise(k, X0, X2) .- pairwise(k, X2, X0)') .< 1e-8)

    # Should be (approximately) positive definite.
    @test all(eigvals(Matrix(pairwise(k, X0))) .> -1e-9)
end

"""
    stationary_kernel_tests(k::Kernel, x0::StepRangeLen, x1::StepRangeLen, x2::StepRangeLen)

Additional tests for stationary kernels. Should be run in addition to `kernel_tests`.
"""
function stationary_kernel_tests(
    k::Kernel,
    x0::StepRangeLen,
    x1::StepRangeLen,
    x2::StepRangeLen,
    x3::StepRangeLen,
    x4::StepRangeLen,
)
    # Check that useful inputs have been provided.
    @assert length(x0) == length(x1)
    @assert length(x0) == length(x2)
    @assert step(x0) == step(x1)
    @assert step(x0) ≠ step(x2)

    @assert length(x3) ≠ length(x0)
    @assert step(x0) == step(x3)

    @assert length(x4) ≠ length(x0)
    @assert step(x4) ≠ length(x0)

    # Unary map.
    @test map(k, x0) isa AbstractFill
    @test map(k, x0) == map(k, collect(x0))

    # Binary map.
    @test map(k, x0, x1) isa AbstractFill
    @test map(k, x0, x1) == map(k, collect(x0), collect(x1))
    @test !isa(map(k, x0, x2), AbstractFill)
    @test map(k, x0, x2) == map(k, collect(x0), collect(x2))

    # Unary pairwise.
    @test pairwise(k, x0) isa SymmetricToeplitz
    @test pairwise(k, x0) == pairwise(k, collect(x0))

    # Binary pairwise.
    @test pairwise(k, x0, x3) isa Toeplitz
    @test pairwise(k, x0, x3) == pairwise(k, collect(x0), collect(x3))
    @test !isa(pairwise(k, x0, x4), Toeplitz)
    @test pairwise(k, x0, x4) == pairwise(k, collect(x0), collect(x4))
end

"""
    differentiable_mean_function_tests(m::MeanFunction, ȳ::AV, x::AV)

Ensure that the gradient w.r.t. the inputs of `MeanFunction` `m` are approximately correct.
"""
function differentiable_mean_function_tests(
    m::MeanFunction,
    ȳ::AbstractVector{<:Real},
    x::AbstractVector{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run forward tests.
    mean_function_tests(m, x)

    # Check adjoint.
    @assert length(ȳ) == length(x)
    adjoint_test(x->map(m, x), ȳ, x; rtol=rtol, atol=atol)
end
function differentiable_mean_function_tests(
    m::MeanFunction,
    ȳ::AbstractVector{<:Real},
    x::ColsAreObs{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run forward tests.
    mean_function_tests(m, x)

    @assert length(ȳ) == length(x)
    adjoint_test(X->map(m, ColsAreObs(X)), ȳ, x.X; rtol=rtol, atol=atol)  
end
function differentiable_mean_function_tests(
    rng::AbstractRNG,
    m::MeanFunction,
    x::AV;
    rtol=_rtol,
    atol=_atol,
)
    return differentiable_mean_function_tests(
        m,
        randn(rng, length(x)),
        x;
        rtol=rtol,
        atol=atol,
    )
end

"""
    differentiable_cross_kernel_tests(
        k::CrossKernel,
        ȳ::AbstractVector{<:Real},
        Ȳ::AbstractMatrix{<:Real},
        x0::AbstractVector,
        x1::AbstractVector,
        x2::AbstractVector,
    )

Ensure that the adjoint w.r.t. the inputs of a `CrossKernel` which is supposed to be
differentiable are approximately correct.
"""
function differentiable_cross_kernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    x0::AbstractVector{<:Real},
    x1::AbstractVector{<:Real},
    x2::AbstractVector{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run forwards-pass cross kernel tests.
    cross_kernel_tests(k, x0, x1, x2)

    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))

    # Binary map.
    adjoint_test(x->map(k, x, x1), ȳ, x0; rtol=rtol, atol=atol)
    adjoint_test(x′->map(k, x0, x′), ȳ, x1; rtol=rtol, atol=atol)

    # Binary pairwise.
    adjoint_test(
        x->reshape(pairwise(k, x, x2), :),
        reshape(Ȳ, :),
        x0;
        rtol=rtol,
        atol=atol,
    )
    adjoint_test(
        x′->reshape(pairwise(k, x0, x′), :),
        reshape(Ȳ, :),
        x2;
        rtol=rtol,
        atol=atol,
    )
end
function differentiable_cross_kernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    x0::ColsAreObs{<:Real},
    x1::ColsAreObs{<:Real},
    x2::ColsAreObs{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run forwards-pass cross kernel tests.
    cross_kernel_tests(k, x0, x1, x2)

    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))

    # Check that dimensionalities are consistent.
    D, N, N′ = size(x0.X, 1), length(x0), length(x2)
    @assert size(x1.X, 1) == D
    @assert size(x2.X, 1) == D

    # Binary map.
    adjoint_test(
        x->map(k, ColsAreObs(reshape(x, div(length(x), length(x1)), :)), x1),
        ȳ,
        reshape(x0.X, :);
        rtol=rtol,
        atol=atol,
    )
    adjoint_test(
        x′->map(k, x0, ColsAreObs(reshape(x′, div(length(x′), length(x0)), :))),
        ȳ,
        reshape(x1.X, :);
        rtol=rtol,
        atol=atol,
    )

    # Binary pairwise.
    adjoint_test(
        x->reshape(pw(k, ColsAreObs(reshape(x, size(x2.X, 1), :)), x2), :),
        reshape(Ȳ, :),
        reshape(x0.X, :);
        rtol=rtol,
        atol=atol,
    )
    adjoint_test(
        x′->reshape(pairwise(k, x0, ColsAreObs(reshape(x′, size(x0.X, 1), :))), :),
        reshape(Ȳ, :),
        reshape(x2.X, :);
        rtol=rtol,
        atol=atol,
    )
end
function differentiable_cross_kernel_tests(
    rng::AbstractRNG,
    k::CrossKernel,
    x0::AV,
    x1::AV,
    x2::AV;
    rtol=_rtol,
    atol=_atol,
)
    ȳ, Ȳ = randn(rng, length(x0)), randn(rng, length(x0), length(x2))
    return differentiable_cross_kernel_tests(k, ȳ, Ȳ, x0, x1, x2; rtol=rtol, atol=atol)
end

"""
    differentiable_kernel_tests(
        k::CrossKernel,
        ȳ::AbstractVector{<:Real},
        Ȳ::AbstractMatrix{<:Real},
        Ȳ_sq::AbstractMatrix{<:Real},
        x0::AbstractVector,
        x1::AbstractVector,
        x2::AbstractVector,
    )

A superset of the tests provided by `differentiable_cross_kernel_tests` designed to test
kernels (which provide unary, in addition to binary, methods for `map` and `pairwise`.)
"""
function differentiable_kernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    Ȳ_sq::AbstractMatrix{<:Real},
    x0::AbstractVector{<:Real},
    x1::AbstractVector{<:Real},
    x2::AbstractVector{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run the forwards-pass kernel tests.
    kernel_tests(k, x0, x1, x2)

    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))
    @assert size(Ȳ_sq, 1) == size(Ȳ_sq, 2)
    @assert size(Ȳ_sq, 1) == length(x0)

    # Run the CrossKernel tests.
    differentiable_cross_kernel_tests(k, ȳ, Ȳ, x0, x1, x2; rtol=rtol, atol=atol)

    # Unary map tests.
    adjoint_test(x->map(k, x), ȳ, x0; rtol=rtol, atol=atol)

    # Unary pairwise test.
    adjoint_test(x->reshape(pairwise(k, x), :), reshape(Ȳ_sq, :), x0; rtol=rtol, atol=atol)
end
function differentiable_kernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    Ȳ_sq::AbstractMatrix{<:Real},
    x0::ColsAreObs{<:Real},
    x1::ColsAreObs{<:Real},
    x2::ColsAreObs{<:Real};
    rtol=_rtol,
    atol=_atol,
)
    # Run the forwards-pass kernel tests.
    kernel_tests(k, x0, x1, x2)

    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))
    @assert size(Ȳ_sq, 1) == size(Ȳ_sq, 2)
    @assert size(Ȳ_sq, 1) == length(x0)

    D, N = size(x0.X)

    # Run the CrossKernel tests.
    differentiable_cross_kernel_tests(k, ȳ, Ȳ, x0, x1, x2; rtol=rtol, atol=atol)

    # Unary map tests.
    adjoint_test(
        x->map(k, ColsAreObs(reshape(x, size(x0.X, 1), :))),
        ȳ,
        reshape(x0.X, :);
        rtol=rtol,
        atol=atol,
    )

    # Unary pairwise test.
    adjoint_test(
        x->reshape(pairwise(k, ColsAreObs(reshape(x, size(x0.X, 1), :))), :),
        reshape(Ȳ_sq, :),
        reshape(x0.X, :);
        rtol=rtol,
        atol=atol,
    )
end
function differentiable_kernel_tests(
    rng::AbstractRNG,
    k::CrossKernel,
    x0::AV,
    x1::AV,
    x2::AV;
    rtol=_rtol,
    atol=_atol,
)
    N, N′ = length(x0), length(x2)
    ȳ, Ȳ, Ȳ_sq = randn(rng, N), randn(rng, N, N′), randn(rng, N, N)
    return differentiable_kernel_tests(k, ȳ, Ȳ, Ȳ_sq, x0, x1, x2; rtol=rtol, atol=atol)
end
