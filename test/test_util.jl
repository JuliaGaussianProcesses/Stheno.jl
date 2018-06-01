import Stheno: AV, AVM, AMRV
using Stheno: unary_obswise, unary_obswise_fallback, binary_obswise,
    binary_obswise_fallback, pairwise, pairwise_fallback, MeanFunction, Kernel, CrossKernel,
    nobs, nfeatures

"""
    unary_obswise_tests(f, X::AVM)

Consistency tests intended for use with `MeanFunction`s.
"""
function unary_obswise_tests(f, X::AVM)
    @test unary_obswise(f, X) isa AbstractVector
    @test length(unary_obswise(f, X)) == nobs(X)
    @test unary_obswise(f, X) ≈ unary_obswise_fallback(f, X)
end

"""
    binary_obswise_tests(f, X::AVM, X′::AVM)

Consistency tests intended for use with `CrossKernel`s.
"""
function binary_obswise_tests(f, X::AVM, X′::AVM)
    @test binary_obswise(f, X, X′) isa AbstractVector
    @test length(binary_obswise(f, X, X′)) == nobs(X)
    @test binary_obswise(f, X, X′) ≈ binary_obswise_fallback(f, X, X′)
end

"""
    binary_obswise_tests(f, X::AVM)

Consistency tests intended for use with `Kernel`s.
"""
function binary_obswise_tests(f, X::AVM)
    @test binary_obswise(f, X) isa AbstractVector
    @test length(binary_obswise(f, X)) == nobs(X)
    @test binary_obswise(f, X) ≈ binary_obswise_fallback(f, X, X)
end

"""
    pairwise_tests(f, X::AVM, X′::AVM)
    

Consistency tests intended for use with `CrossKernel`s.
"""
function pairwise_tests(f, X::AVM, X′::AVM)
    @test pairwise(f, X, X′) isa AbstractMatrix
    @test size(pairwise(f, X, X′)) == (nobs(X), nobs(X′))
    @test pairwise(f, X, X′) ≈ pairwise_fallback(f, X, X′)
end

"""
    pairwise_tests(f, X::AVM)

Consistency tests intended for use with `Kernel`s.
"""
function pairwise_tests(f, X::AVM)
    @test pairwise(f, X) isa AbstractMatrix
    @test size(pairwise(f, X)) == (nobs(X), nobs(X))
    @test pairwise(f, X) ≈ pairwise_fallback(f, X, X)
end

"""
    mean_function_tests(μ::MeanFunction, X::AVM)

Tests that any mean function `μ` should be able to pass.
"""
function mean_function_tests(μ::MeanFunction, X::AVM)
    unary_obswise_tests(μ, X)
end

"""
    cross_kernel_tests(k::CrossKernel, X::AVM, X′::AVM)

Tests that any kernel `k` should be able to pass. Requires that `nobs(X0) == nobs(X1)` and
`nobs(X0) ≠ nobs(X2)`.
"""
function cross_kernel_tests(k::CrossKernel, X0::AVM, X1::AVM, X2::AVM)
    @assert nobs(X0) == nobs(X1)
    @assert nobs(X0) ≠ nobs(X2)

    binary_obswise_tests(k, X0, X1)
    pairwise_tests(k, X0, X2)
end

"""
    kernel_tests(k::Kernel, X0::AVM, X1::AVM, X2::AVM)

Tests that any kernel `k` should be able to pass. Requires that `nobs(X0) == nobs(X1)` and
`nobs(X0) ≠ nobs(X2)`.
"""
function kernel_tests(k::Kernel, X0::AVM, X1::AVM, X2::AVM)
    @assert nobs(X0) == nobs(X1)
    @assert nobs(X0) ≠ nobs(X2)

    # Generic tests.
    cross_kernel_tests(k, X0, X1, X2)
    binary_obswise(k, X0)
    pairwise_tests(k, X0)

    # Kernels should be symmetric for same arguments.
    @test pairwise(k, X0) ≈ pairwise(k, X0)'
    @test pairwise(k, X0) isa LazyPDMat

    # k(x, x′) == k(x′, x)
    @test binary_obswise(k, X0, X1) ≈ binary_obswise(k, X1, X0)
    @test pairwise(k, X0, X2) ≈ pairwise(k, X2, X0)'

    # Should be (approximately) positive definite.
    @test all(eigvals(Matrix(pairwise(k, X0))) .> -1e-9)
end
