using IterTools
using Stheno: MeanFunction, Kernel, CrossKernel, AVM

"""
    unary_map_tests(f, X::AbstractDataSet)

Consistency tests intended for use with `MeanFunction`s.
"""
function unary_map_tests(f, X::ADS)
    @test map(f, X) isa AbstractVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ [f(x) for x in X]
end

"""
    binary_map_tests(f, X::ADS, X′::ADS)

Consistency tests intended for use with `CrossKernel`s.
"""
function binary_map_tests(f, X::ADS, X′::ADS)
    @test map(f, X, X′) isa AbstractVector
    @test length(map(f, X, X′)) == length(X)
    @test map(f, X, X′) ≈ [f(x, x′) for (x, x′) in zip(X, X′)]
end

"""
    binary_map_tests(f, X::ADS)

Consistency tests intended for use with `Kernel`s.
"""
function binary_map_tests(f, X::ADS)
    @test map(f, X) isa AbstractVector
    @test length(map(f, X)) == length(X)
    @test map(f, X) ≈ map(f, X, X)
end

"""
    pairwise_tests(f, X::ADS, X′::ADS)

Consistency tests intended for use with `CrossKernel`s.
"""
function pairwise_tests(f, X::ADS, X′::ADS)
    N, N′ = length(X), length(X′)
    @test pairwise(f, X, X′) isa AbstractMatrix
    @test size(pairwise(f, X, X′)) == (N, N′)
    @test pairwise(f, X, X′) ≈ reshape([f(x, x′) for (x, x′) in product(X, X′)], N, N′)
end

"""
    pairwise_tests(f, X::ADS)

Consistency tests intended for use with `Kernel`s.
"""
function pairwise_tests(f, X::ADS; rtol=eps())
    @test pairwise(f, X) isa AbstractMatrix
    @test size(pairwise(f, X)) == (length(X), length(X))
    @test isapprox(pairwise(f, X), pairwise(f, X, X); rtol=rtol)
end

"""
    mean_function_tests(μ::MeanFunction, X::ADS)

Tests that any mean function `μ` should be able to pass.
"""
function mean_function_tests(μ::MeanFunction, X::ADS)

    # Test compulsory interface passes.
    @test method_exists(μ, Tuple{eltype(X)})
    @test !(μ(X[1]) isa Void)

    @test method_exists(eachindex, Tuple{typeof(μ)})

    # Test optional interface.
    unary_map_tests(μ, X)
end
mean_function_tests(μ::MeanFunction, X::AVM{<:Real}) = mean_function_tests(μ, DataSet(X))

"""
    cross_kernel_tests(k::CrossKernel, X::ADS, X′::ADS)

Tests that any kernel `k` should be able to pass. Requires that `length(X0) == length(X1)`
and `length(X0) ≠ length(X2)`.
"""
function cross_kernel_tests(k::CrossKernel, X0::ADS, X1::ADS, X2::ADS)
    @assert length(X0) == length(X1)
    @assert length(X0) ≠ length(X2)

    binary_map_tests(k, X0, X1)
    pairwise_tests(k, X0, X2)
end
function cross_kernel_tests(
    k::CrossKernel,
    X0::AVM{<:Real},
    X1::AVM{<:Real},
    X2::AVM{<:Real},
)
    return cross_kernel_tests(k, DataSet(X0), DataSet(X1), DataSet(X2))
end

"""
    kernel_tests(k::Kernel, X0::ADS, X1::ADS, X2::ADS)

Tests that any kernel `k` should be able to pass. Requires that `length(X0) == length(X1)`
and `length(X0) ≠ length(X2)`.
"""
function kernel_tests(k::Kernel, X0::ADS, X1::ADS, X2::ADS, rtol::Real=eps())
    @assert length(X0) == length(X1)
    @assert length(X0) ≠ length(X2)

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
end
function kernel_tests(
    k::Kernel,
    X0::AVM{<:Real},
    X1::AVM{<:Real},
    X2::AVM{<:Real},
    rtol::Real=eps(),
)
   return kernel_tests(k, DataSet(X0), DataSet(X1), DataSet(X2), rtol)
end
