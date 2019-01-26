using BlockArrays, LinearAlgebra, FDM, Zygote, ToeplitzMatrices
using Stheno: MeanFunction, Kernel, CrossKernel, AV, pairwise
using FillArrays: AbstractFill, getindex_value
using LinearAlgebra: AbstractTriangular

const _rtol = 1e-10
const _atol = 1e-10

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)
    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println()
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

# Transform `x` into a vector, and return a closure which inverts the transformation.
to_vec(x::Nothing) = (x, nothing)
to_vec(x::Real) = ([x], x->x[1])

# Arrays.
to_vec(x::Vector{<:Real}) = (x, identity)
to_vec(x::Array) = vec(x), x_vec->reshape(x_vec, size(x))

# AbstractArrays.
to_vec(x::ColsAreObs{<:Real}) = (vec(x.X), x_vec->ColsAreObs(reshape(x_vec, size(x.X))))
to_vec(x::BlockArray) = vec(Array(x)), x_->BlockArray(reshape(x_, size(x)), blocksizes(x))
to_vec(x::AbstractFill) = vec(x), nothing
function to_vec(x::T) where {T<:AbstractTriangular}
    x_vec, back = to_vec(Matrix(x))
    return x_vec, x_vec->T(reshape(back(x_vec), size(x)))
end
to_vec(x::Symmetric) = vec(Matrix(x)), x_vec->Symmetric(reshape(x_vec, size(x)))


# Non-array data structures.
function to_vec(x::Tuple)
    x_vecs, x_backs = zip(map(to_vec, x)...)
    sz = cumsum([map(length, x_vecs)...])
    return vcat(x...), v->([x_backs[n](v[sz[n]-length(x[n])+1:sz[n]]) for n in 1:length(x)]...,)
end

function FDM.j′vp(fdm, f, ȳ, x)
    x_vec, vec_to_x = to_vec(x)
    ȳ_vec, _ = to_vec(ȳ)
    return vec_to_x(FDM.j′vp(fdm, x_vec->to_vec(f(vec_to_x(x_vec)))[1], ȳ_vec, x_vec))
end

function j′vp(fdm, f, ȳ, xs...)
    return (map(enumerate(xs)) do (p, x)
        return FDM.j′vp(
            fdm,
            function(x)
                xs_ = [xs...]
                xs_[p] = x
                return f(xs_...)
            end,
            ȳ,
            x,
        )    
    end...,)
end

# My version of isapprox
function fd_isapprox(x_ad::Nothing, x_fd, rtol, atol)
    return isapprox(x_fd, zero(x_fd); rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::AbstractArray, x_fd::AbstractArray, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::Real, x_fd::Real, rtol, atol)
    return isapprox(x_ad, x_fd; rtol=rtol, atol=atol)
end
function fd_isapprox(x_ad::NamedTuple, x_fd, rtol, atol)
    f = (x_ad, x_fd)->fd_isapprox(x_ad, x_fd, rtol, atol)
    return all([f(getfield(x_ad, key), getfield(x_fd, key)) for key in keys(x_ad)])
end
function fd_isapprox(x_ad::Tuple, x_fd::Tuple, rtol, atol)
    return all(map((x, x′)->fd_isapprox(x, x′, rtol, atol), x_ad, x_fd))
end

# Ensure that `to_vec` and j′vp works correctly.
for x in [randn(10), 5.0, randn(10, 10), ColsAreObs(randn(2, 5)), (5.0, 4.0),
        (randn(10), randn(11)), BlockVector([randn(10)]),]
    x_vec, back = to_vec(x)
    @test x_vec isa AbstractVector{<:Real}
    @test back(x_vec) == x
    @test fd_isapprox(j′vp(central_fdm(5, 1), identity, x, x), (x,), 1e-10, 1e-10)
end

# Ensure that forwards- and reverse- passes are (approximately) correct.
function adjoint_test(f, ȳ, x...; rtol=_rtol, atol=_atol, fdm=central_fdm(5, 1))

    # Compute forwards-pass and j′vp.
    y, back = Zygote.forward(f, x...)
    adj_ad = back(ȳ)
    adj_fd = j′vp(fdm, f, ȳ, x...)

    # Check that forwards-pass agrees with plain forwards-pass.
    @test y == f(x...)

    # Check that ad and fd adjoints (approximately) agree.
    # print_adjoints(adj_ad, adj_fd, rtol, atol)
    @test fd_isapprox(adj_ad, adj_fd, rtol, atol)
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
# function mean_function_tests(μ::MeanFunction, X::BlockData)
#     __mean_function_tests(μ, X)
# end
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
# function cross_kernel_tests(k::CrossKernel, X0::BlockData, X1::BlockData, X2::BlockData)
#     __cross_kernel_tests(k, X0, X1, X2)
# end
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
    x0::AbstractVector,
    x1::AbstractVector,
    x2::AbstractVector;
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
    adjoint_test((x, x′)->map(k, x, x′), ȳ, x0, x1; rtol=rtol, atol=atol)

    # Binary pairwise.
    adjoint_test((x, x′)->pw(k, x, x′), Ȳ, x0, x2; rtol=rtol, atol=atol)
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
    x0::AbstractVector,
    x1::AbstractVector,
    x2::AbstractVector;
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
    adjoint_test(x->pw(k, x), Ȳ_sq, x0; rtol=rtol, atol=atol)
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
