using BlockArrays, LinearAlgebra, FDM, Zygote, ToeplitzMatrices
using Stheno: MeanFunction, Kernel, CrossKernel, AV, pairwise, pw, BlockData, blocks
using FillArrays: AbstractFill, getindex_value
using LinearAlgebra: AbstractTriangular
using FDM: j′vp
import FDM: to_vec

const _rtol = 1e-10
const _atol = 1e-10

Base.length(::Nothing) = 0

function print_adjoints(adjoint_ad, adjoint_fd, rtol, atol)
    @show typeof(adjoint_ad), typeof(adjoint_fd)
    adjoint_ad, adjoint_fd = to_vec(adjoint_ad)[1], to_vec(adjoint_fd)[1]
    println("atol is $atol, rtol is $rtol")
    println("ad, fd, abs, rel")
    abs_err = abs.(adjoint_ad .- adjoint_fd)
    rel_err = abs_err ./ adjoint_ad
    display([adjoint_ad adjoint_fd abs_err rel_err])
    println()
end

# AbstractArrays.
to_vec(x::ColsAreObs{<:Real}) = vec(x.X), x_vec -> ColsAreObs(reshape(x_vec, size(x.X)))
to_vec(x::BlockArray) = vec(Array(x)), x_->BlockArray(reshape(x_, size(x)), blocksizes(x))
to_vec(x::AbstractFill) = vec(Array(x)), x->error("No backwards defined")
function to_vec(x::BlockData)
    x_vecs, x_backs = zip(map(to_vec, blocks(x))...)
    sz = cumsum([map(length, x_vecs)...])
    return vcat(x_vecs...), function(v)
        return BlockData([x_backs[n](v[sz[n]-length(blocks(x)[n])+1:sz[n]])
            for n in 1:length(blocks(x))])
    end
end
function to_vec(X::T) where T<:Union{Adjoint,Transpose}
    U = T.name.wrapper
    return vec(Matrix(X)), x_vec->U(permutedims(reshape(x_vec, size(X))))
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

function adjoint_test(f, ȳ, x...; rtol=_rtol, atol=_atol, fdm=central_fdm(5, 1))

    # Compute forwards-pass and j′vp.
    y, back = Zygote.forward(f, x...)
    adj_ad = back(ȳ)
    adj_fd = j′vp(fdm, f, ȳ, x...)

    # If unary, pull out first thing from ad.
    adj_ad = length(x) == 1 ? first(adj_ad) : adj_ad

    # Check that forwards-pass agrees with plain forwards-pass.
    @test y ≈ f(x...)

    # Check that ad and fd adjoints (approximately) agree.
    # print_adjoints(adj_ad, adj_fd, rtol, atol)
    @test fd_isapprox(adj_ad, adj_fd, rtol, atol)
end

"""
    mean_function_tests(m::MeanFunction, X::AbstractVector)

Test _very_ basic consistency properties of the mean function `m`.
"""
function mean_function_tests(m::MeanFunction, x::AbstractVector)
    @test map(m, x) isa AbstractVector
    @test length(map(m, x)) == length(x)
end

"""
    cross_kernel_tests(k::CrossKernel, x0::AV, x1::AV, x2::AV)

Tests that any cross kernel `k` should be able to pass. Requires that
`length(x0) == length(x1)` and `length(x0) ≠ length(x2)`.
"""
function cross_kernel_tests(k::CrossKernel, x0::AV, x1::AV, x2::AV; atol=1e-9)
    @assert length(x0) == length(x1)
    @assert length(x0) ≠ length(x2)

    # Check that map basically works.
    @test map(k, x0, x1) isa AbstractVector
    @test length(map(k, x0, x1)) == length(x0)

    # Check that pairwise basically works.
    @test pairwise(k, x0, x2) isa AbstractMatrix
    @test size(pairwise(k, x0, x2)) == (length(x0), length(x2))

    # Check that map is consistent with pairwise.
    @test map(k, x0, x1) ≈ diag(pairwise(k, x0, x1)) atol=atol
end

"""
    kernel_tests(k::Kernel, X0::AbstractVector, X1::AbstractVector, X2::AbstractVector)

Tests that any kernel `k` should be able to pass. Requires that `length(X0) == length(X1)`
and `length(X0) ≠ length(X2)`.
"""
function kernel_tests(k::Kernel, x0::AV, x1::AV, x2::AV; atol=1e-9)
    @assert length(x0) == length(x1)
    @assert length(x0) ≠ length(x2)

    # Check that all of the binary methods work as expected.
    cross_kernel_tests(k, x0, x1, x2)

    # Check additional binary map properties for kernels.
    @test map(k, x0, x1) ≈ map(k, x1, x0)
    @test pairwise(k, x0, x2) ≈ pairwise(k, x2, x0)' atol=atol

    # Check that unary map basically works.
    @test map(k, x0) isa AbstractVector
    @test length(map(k, x0)) == length(x0)

    # Check that unary pairwise basically works.
    @test pairwise(k, x0) isa AbstractMatrix
    @test size(pairwise(k, x0)) == (length(x0), length(x0))
    @test pairwise(k, x0) ≈ pairwise(k, x0)' atol=atol

    # Check that unary map is consistent with unary pairwise.
    @test map(k, x0) ≈ diag(pairwise(k, x0)) atol=atol

    # Check that unary pairwise produces a positive definite matrix (approximately).
    @test all(eigvals(Matrix(pairwise(k, x0))) .> -atol)

    # Check that unary map / pairwise are consistent with the binary versions.
    @test map(k, x0) ≈ map(k, x0, x0) atol=atol
    @test pairwise(k, x0) ≈ pairwise(k, x0, x0) atol=atol
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
