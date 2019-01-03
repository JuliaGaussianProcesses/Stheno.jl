using BlockArrays, LinearAlgebra, FDM, Zygote, ToeplitzMatrices
using Stheno: MeanFunction, Kernel, CrossKernel, AV, blocks, pairwise
using FillArrays: AbstractFill

function grad_test(f, x::AbstractVector)
    grad_ad = Zygote.gradient(f, x)[1]
    grad_fd = FDM.grad(central_fdm(5, 1), f, x)
    @test maximum(abs.(grad_ad .- grad_fd)) < 1e-8
end
function grad_test(f, x::ColsAreObs)
    grad_ad = Zygote.gradient(X->f(ColsAreObs(X)), x.X)[1]
    grad_fd = FDM.grad(central_fdm(5, 1), X->f(ColsAreObs(X)), x.X)
    @test maximum(abs.(grad_ad .- grad_fd)) < 1e-8
end

"""
    unary_map_tests(f, X::AbstractVector)

Consistency tests intended for use with `MeanFunction`s.
"""
function unary_map_tests(f, X::AbstractVector)
    @test map(f, X) isa AbstractVector
    @test length(f.(X)) == length(X)
    @test map(f, X) ≈ [f(x) for x in X]
    grad_test(x->sum(map(f, x)), X)
end
# function unary_map_tests(f, X::BlockData)
#     @test map(f, X) isa AbstractBlockVector
#     @test length(map(f, X)) == length(X)
#     @test map(f, X) ≈ BlockVector([map(f, x) for x in blocks(X)])
# end

"""
    binary_map_tests(f, X::AbstractVector, X′::AbstractVector)

Consistency tests intended for use with `CrossKernel`s.
"""
function binary_map_tests(f, X::AbstractVector, X′::AbstractVector)
    @test map(f, X, X′) isa AbstractVector
    @test length(map(f, X, X′)) == length(X)
    @test map(f, X, X′) ≈ [f(x, x′) for (x, x′) in zip(X, X′)]
    grad_test(x->sum(map(f, x, X′)), X)
    grad_test(x′->sum(map(f, X, x′)), X′)
end
# function binary_map_tests(f, XB::BlockData, XB′::BlockData)
#     @test map(f, XB, XB′) isa AbstractBlockVector
#     @test length(map(f, XB, XB′)) == length(XB)
#     @test map(f, XB, XB′) ≈
#         BlockVector([map(f, X, X′) for (X, X′) in zip(blocks(XB), blocks(XB′))])
# end

"""
    binary_map_tests(f, X::AbstractVector)

Consistency tests intended for use with `Kernel`s.
"""
function binary_map_tests(f, X::AbstractVector)
    @test map(f, X) isa AbstractVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ map(f, X, X)
    grad_test(x->sum(map(f, x)), X)
end
# function binary_map_tests(f, X::BlockData)
#     @test map(f, X) isa AbstractBlockVector
#     @test length(map(f, X)) == length(X)
#     @test map(f, X) ≈ map(f, X, X)
# end

"""
    pairwise_tests(f, X::AV, X′::AV)

Consistency tests intended for use with `CrossKernel`s.
"""
function pairwise_tests(f, X::AbstractVector, X′::AbstractVector)
    N, N′ = length(X), length(X′)
    @test pairwise(f, X, X′) isa AbstractMatrix
    @test size(pairwise(f, X, X′)) == (N, N′)
    @test pairwise(f, X, X′) ≈ reshape([f(x, x′) for (x, x′) in Iterators.product(X, X′)], N, N′)
    grad_test(x->sum(pairwise(f, x, X′)), X)
    grad_test(x′->sum(pairwise(f, X, x′)), X′)
end
# function pairwise_tests(f, X::BlockData, X′::BlockData)
#     N, N′ = length(X), length(X′)
#     @test pairwise(f, X, X′) isa AbstractBlockMatrix
#     @test size(pairwise(f, X, X′)) == (N, N′)
#     @test pairwise(f, X, X′) ==
#         BlockMatrix([pairwise(f, x, x′) for x in blocks(X), x′ in blocks(X′)])
# end

"""
    pairwise_tests(f, X::AV)

Consistency tests intended for use with `Kernel`s.
"""
function pairwise_tests(f, X::AbstractVector; rtol=eps())
    @test pairwise(f, X) isa AbstractMatrix
    @test size(pairwise(f, X)) == (length(X), length(X))
    @test isapprox(pairwise(f, X), pairwise(f, X, X); rtol=rtol)
end
# function pairwise_tests(f, X::BlockData; rtol=eps())
#     @test size(pairwise(f, X)) == (length(X), length(X))
#     @test pairwise(f, X) ==
#         BlockMatrix([pairwise(f, x, x′) for x in blocks(X), x′ in blocks(X)])
# end

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
    # kernel_tests(k, BlockData([X0, X0]), BlockData([X1, X1]), BlockData([X2]), rtol)
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
    @test pairwise(k, X0, X2) ≈ pairwise(k, X2, X0)'

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
