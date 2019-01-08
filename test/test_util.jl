using BlockArrays, LinearAlgebra, FDM, Zygote, ToeplitzMatrices
using Stheno: MeanFunction, Kernel, CrossKernel, AV, blocks, pairwise
using FillArrays: AbstractFill

# Compare FDM estimate of the adjoint with Zygotes. Also ensure that forwards passes match.
function adjoint_test(f, ȳ::AbstractVector{<:Real}, x::AbstractVector{<:Real})

    # Compute Zygote forward-pass and ensure that it matches regular evaulation of `f`.
    y, back = Zygote.forward(f, x)
    @test y == f(x)

    # Compare gradients.
    adjoint_ad = back(ȳ)[1]
    adjoint_fd = FDM.adjoint(central_fdm(5, 1), f, ȳ, x)
    @test (adjoint_ad isa Nothing &&
        isapprox(adjoint_fd, zero(adjoint_fd); rtol=1e-5, atol=1e5)) ||
        isapprox(adjoint_ad, adjoint_fd; rtol=1e-5, atol=1e-5)
end
adjoint_test(f, ȳ::Real, x::AbstractVector{<:Real}) = adjoint_test(x->[f(x)], [ȳ], x)
adjoint_test(f, ȳ::AbstractVector{<:Real}, x::Real) = adjoint_test(x->f(x[1]), ȳ, [x])
adjoint_test(f, ȳ::Real, x::Real) = adjoint_test(x->[f(x[1])], [ȳ], [x])

# function grad_test(f, x::AbstractVector)
#     grad_ad = Zygote.gradient(f, x)[1]
#     grad_fd = FDM.grad(central_fdm(5, 1), f, x)
#     @test (grad_ad isa Nothing && all(abs.(grad_fd) .< 1e-8)) ||
#         maximum(abs.(grad_ad .- grad_fd)) < 1e-6
# end
# function grad_test(f, x::ColsAreObs)
#     grad_ad = Zygote.gradient(X->f(ColsAreObs(X)), x.X)[1]
#     grad_fd = FDM.grad(central_fdm(5, 1), X->f(ColsAreObs(X)), x.X)
#     println("ad")
#     display(grad_ad)
#     println()
#     println("fd")
#     display(grad_fd)
#     println()

#     if !(grad_ad isa Nothing)
#         println("diff")
#         display(grad_ad .- grad_fd)
#         println()
#     end

#     @test (grad_ad isa Nothing && all(abs.(grad_fd) .< 1e-8)) ||
#         maximum(abs.(grad_ad .- grad_fd)) < 1e-6
# end

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
    differentiable_crosskernel_tests(
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
function differentiable_crosskernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    x0::AbstractVector{<:Real},
    x1::AbstractVector{<:Real},
    x2::AbstractVector{<:Real},
)
    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))

    # Binary map.
    adjoint_test(x->map(k, x, x1), ȳ, x0)
    adjoint_test(x′->map(k, x0, x′), ȳ, x1)

    # Binary pairwise.
    adjoint_test(x->reshape(pairwise(k, x, x2), :), reshape(Ȳ, :), x0)
    adjoint_test(x′->reshape(pairwise(k, x0, x′), :), reshape(Ȳ, :), x2)
end
function differentiable_crosskernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    x0::ColsAreObs{<:Real},
    x1::ColsAreObs{<:Real},
    x2::ColsAreObs{<:Real},
)
    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))

    # Check that dimensionalities are consistent.
    D, N, N′ = size(x0.X, 1), length(x0), length(x2)
    @assert size(x1.X, 1) == D
    @assert size(x2.X, 1) == D

    # Binary map.
    adjoint_test(x->map(k, ColsAreObs(reshape(x, D, N)), x1), ȳ, reshape(x0.X, :))
    adjoint_test(x′->map(k, x0, ColsAreObs(reshape(x′, D, N))), ȳ, reshape(x1.X, :))

    # Binary pairwise.
    adjoint_test(
        x->reshape(pairwise(k, ColsAreObs(reshape(x, D, N)), x2), :),
        reshape(Ȳ, :),
        reshape(x0.X, :),
    )
    adjoint_test(
        x′->reshape(pairwise(k, x0, ColsAreObs(reshape(x′, D, N′))), :),
        reshape(Ȳ, :),
        reshape(x2.X, :),
    )
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

A superset of the tests provided by `differentiable_crosskernel_tests` designed to test
kernels (which provide unary, in addition to binary, methods for `map` and `pairwise`.)
"""
function differentiable_kernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    Ȳ_sq::AbstractMatrix{<:Real},
    x0::AbstractVector{<:Real},
    x1::AbstractVector{<:Real},
    x2::AbstractVector{<:Real},
)
    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))
    @assert size(Ȳ_sq, 1) == size(Ȳ_sq, 2)
    @assert size(Ȳ_sq, 1) == length(x0)

    # Run the CrossKernel tests.
    differentiable_crosskernel_tests(k, ȳ, Ȳ, x0, x1, x2)

    # Unary map tests.
    adjoint_test(x->map(k, x), ȳ, x0)

    # Unary pairwise test.
    adjoint_test(x->reshape(pairwise(k, x), :), reshape(Ȳ_sq, :), x0)
end
function differentiable_kernel_tests(
    k::CrossKernel,
    ȳ::AbstractVector{<:Real},
    Ȳ::AbstractMatrix{<:Real},
    Ȳ_sq::AbstractMatrix{<:Real},
    x0::ColsAreObs{<:Real},
    x1::ColsAreObs{<:Real},
    x2::ColsAreObs{<:Real},
)
    # Ensure that the inputs are as required.
    @assert length(ȳ) == length(x0)
    @assert length(ȳ) == length(x1)
    @assert size(Ȳ) == (length(x0), length(x2))
    @assert size(Ȳ_sq, 1) == size(Ȳ_sq, 2)
    @assert size(Ȳ_sq, 1) == length(x0)

    D, N = size(x0.X)

    # Run the CrossKernel tests.
    differentiable_crosskernel_tests(k, ȳ, Ȳ, x0, x1, x2)

    # Unary map tests.
    adjoint_test(x->map(k, ColsAreObs(reshape(x, D, N))), ȳ, reshape(x0.X, :))

    # Unary pairwise test.
    adjoint_test(
        x->reshape(pairwise(k, ColsAreObs(reshape(x, D, N))), :),
        reshape(Ȳ_sq, :),
        reshape(x0.X, :),
    )
end
