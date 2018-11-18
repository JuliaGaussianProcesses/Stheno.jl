using BlockArrays, LinearAlgebra
using Stheno: MeanFunction, Kernel, CrossKernel, AV, blocks, pairwise, LazyPDMat

"""
    unary_map_tests(f, X::AbstractVector)

Consistency tests intended for use with `MeanFunction`s.
"""
function unary_map_tests(f, X::AbstractVector)
    @test map(f, X) isa AbstractVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ [f(x) for x in X]
    @test map(f, X) ≈ Stheno._map_fallback(f, X)
end
function unary_map_tests(f, X::BlockData)
    @test map(f, X) isa AbstractBlockVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ BlockVector([map(f, x) for x in blocks(X)])
end

"""
    binary_map_tests(f, X::AbstractVector, X′::AbstractVector)

Consistency tests intended for use with `CrossKernel`s.
"""
function binary_map_tests(f, X::AbstractVector, X′::AbstractVector)
    @test map(f, X, X′) isa AbstractVector
    @test length(map(f, X, X′)) == length(X)
    @test map(f, X, X′) ≈ [f(x, x′) for (x, x′) in zip(X, X′)]
    @test map(f, X, X′) ≈ Stheno._map_fallback(f, X, X′)
end
function binary_map_tests(f, XB::BlockData, XB′::BlockData)
    @test map(f, XB, XB′) isa AbstractBlockVector
    @test length(map(f, XB, XB′)) == length(XB)
    @test map(f, XB, XB′) ≈
        BlockVector([map(f, X, X′) for (X, X′) in zip(blocks(XB), blocks(XB′))])
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
function binary_map_tests(f, X::BlockData)
    @test map(f, X) isa AbstractBlockVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ map(f, X, X)
end

"""
    pairwise_tests(f, X::ADS, X′::ADS)

Consistency tests intended for use with `CrossKernel`s.
"""
function pairwise_tests(f, X::AbstractVector, X′::AbstractVector)
    N, N′ = length(X), length(X′)
    @test pairwise(f, X, X′) isa AbstractMatrix
    @test size(pairwise(f, X, X′)) == (N, N′)
    @test pairwise(f, X, X′) ≈ reshape([f(x, x′) for (x, x′) in Iterators.product(X, X′)], N, N′)
    @test pairwise(f, X, X′) ≈ Stheno._pairwise_fallback(f, X, X′)
end
function pairwise_tests(f, X::BlockData, X′::BlockData)
    N, N′ = length(X), length(X′)
    @test pairwise(f, X, X′) isa AbstractBlockMatrix
    @test size(pairwise(f, X, X′)) == (N, N′)
    @test pairwise(f, X, X′) ==
        BlockMatrix([pairwise(f, x, x′) for x in blocks(X), x′ in blocks(X′)])
end

"""
    pairwise_tests(f, X::ADS)

Consistency tests intended for use with `Kernel`s.
"""
function pairwise_tests(f, X::AbstractVector; rtol=eps())
    @test pairwise(f, X) isa LazyPDMat{T, <:AbstractMatrix{T}} where T
    @test size(pairwise(f, X)) == (length(X), length(X))
    @test isapprox(pairwise(f, X), pairwise(f, X, X); rtol=rtol)
end
function pairwise_tests(f, X::BlockData; rtol=eps())
    @test pairwise(f, X) isa LazyPDMat{T, <:Symmetric{T, <:AbstractBlockMatrix{T}}} where T
    @test size(pairwise(f, X)) == (length(X), length(X))
    @test pairwise(f, X) ==
        BlockMatrix([pairwise(f, x, x′) for x in blocks(X), x′ in blocks(X)])
end

"""
    mean_function_tests(μ::MeanFunction, X::AbstractVector)

Tests that any mean function `μ` should be able to pass.
"""
function mean_function_tests(μ::MeanFunction, X::AbstractVector)
    __mean_function_tests(μ, X)
    mean_function_tests(μ, BlockData([X]))
    mean_function_tests(μ, BlockData([X, X]))
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
    cross_kernel_tests(k, BlockData([X0, X0]), BlockData([X1, X1]), BlockData([X2]))
end
function cross_kernel_tests(k::CrossKernel, X0::BlockData, X1::BlockData, X2::BlockData)
    __cross_kernel_tests(k, X0, X1, X2)
end
function __cross_kernel_tests(k::CrossKernel, X0::AV, X1::AV, X2::AV)
    @assert length(X0) == length(X1)
    @assert length(X0) ≠ length(X2)

    @test hasmethod(eachindex, Tuple{typeof(k), Int})

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
    kernel_tests(k, BlockData([X0, X0]), BlockData([X1, X1]), BlockData([X2]), rtol)
end
function kernel_tests(k::Kernel, X0::AV, X1::AV, X2::AV, rtol::Real=eps())
    __kernel_tests(k, X0, X1, X2, rtol)
end
function __kernel_tests(k::Kernel, X0::AV, X1::AV, X2::AV, rtol::Real=eps())
    @assert length(X0) == length(X1)
    @assert length(X0) ≠ length(X2)

    @test hasmethod(eachindex, Tuple{typeof(k)})

    # Generic tests.
    cross_kernel_tests(k, X0, X1, X2)
    binary_map_tests(k, X0)
    pairwise_tests(k, X0; rtol=rtol)

    # Kernels should be symmetric for same arguments.
    @test pairwise(k, X0) isa LazyPDMat
    @test pairwise(k, X0) ≈ pairwise(k, X0)'

    # k(x, x′) == k(x′, x)
    @test map(k, X0, X1) ≈ map(k, X1, X0)
    @test pairwise(k, X0, X2) ≈ pairwise(k, X2, X0)'

    # Should be (approximately) positive definite.
    @test all(eigvals(Matrix(pairwise(k, X0))) .> -1e-9)

    # length should equal length of first side.
    @test length(k) == size(k, 1)
    @test size(k, 1) == size(k, 2)
end
