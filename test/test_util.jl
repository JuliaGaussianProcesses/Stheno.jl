import Stheno: AV, AVM

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
