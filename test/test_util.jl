import Stheno: AV, AVM, AMRV
using Stheno: unary_colwise, unary_colwise_fallback, binary_colwise,
    binary_colwise_fallback, pairwise, pairwise_fallback

# A collection of tests that any kernel should definitely be able to pass.
function _generic_kernel_tests(k::Kernel, X::AVM, X′::AVM)
    N, N′ = sum(size.(X[:, 1], 1)), sum(size.(X′[:, 1], 1))

    # Test mechanics.
    @test k == k
    @test size(xcov(k, X, X)) == (N, N)
    @test size(xcov(k, X, X′)) == (N, N′)
    @test size(xcov(k, X′, X)) == (N′, N)
    @test typeof(cov(k, X)) <: LazyPDMat
    @test typeof(marginal_cov(k, X)) <: AbstractVector

    # Test numerical output consistency.
    @test xcov(k, X, X) ≈ cov(k, X)
    @test marginal_cov(k, X) ≈ diag(Matrix(cov(k, X)))
    @test xcov(k, X, X′) ≈ xcov(k, X′, X)'
end

"""
    unary_colwise_tests(f, X::AMRV)

Check that definitions of `unary_colwise` are self-consistent for `f`. Requires both `x` and
`X` to ensure that optimisations for single input and scalar-valued element cases are
handled correctly.
"""
function unary_colwise_tests(f, X::AMRV)
    @test unary_colwise(f, X) isa AbstractVector
    @test length(unary_colwise(f, X)) == size(X, 2)
    @test unary_colwise(f, X) ≈ unary_colwise_fallback(f, X)
end

"""
    binary_colwise_tests(f, X::AMRV, X′::AMRV)

Check that definitions of `binary_colwise` are self-consistent for `f`.
"""
function binary_colwise_tests(f, X::AMRV, X′::AMRV)
    
    @assert size(X) == size(X′)

    @test binary_colwise(f, X) isa AbstractVector
    @test binary_colwise(f, X, X′) isa AbstractVector
    @test length(binary_colwise(f, X)) == size(X, 2)
    @test length(binary_colwise(f, X, X′)) == size(X, 2)

    @test binary_colwise(f, X) ≈ binary_colwise_fallback(f, X, X)
    @test binary_colwise(f, X, X′) ≈ binary_colwise_fallback(f, X, X′)
end

"""
    pairwise_tests(f, X::AMRV, X′::AMRV)

Check that the definitions of `pairwise` are self-consistent for `f`.
"""
function pairwise_tests(f, X::AMRV, X′::AMRV)

    @assert size(X, 1) == size(X′, 1)

    @test pairwise(f, X) isa AbstractMatrix
    @test pairwise(f, X, X′) isa AbstractMatrix
    @test size(pairwise(f, X)) == (size(X, 2), size(X, 2))
    @test size(pairwise(f, X, X′)) == (size(X, 2), size(X′, 2))

    @test pairwise(f, X) ≈ pairwise_fallback(f, X, X)
    @test pairwise(f, X, X′) ≈ pairwise_fallback(f, X, X′)
end

"""
    mean_function_tests(f, X::AMRV)

Tests that any mean function `f` should be able to pass.
"""
function mean_function_tests(f, X::AMRV)
    unary_colwise_tests(f, X)
end

"""
    kernel_tests(k, X::AMRV)

Tests that any kernel `k` should be able to pass.
"""
function kernel_tests(k, X::AMRV, X′::AMRV)
    binary_colwise_tests(k, X, X′)
    pairwise_tests(k, X, X′)
end
