import Base: eachindex, AbstractVector, AbstractMatrix
export FiniteMean, FiniteKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    FiniteCrossKernel

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean <: MeanFunction
    μ::MeanFunction
    X::AVM
end
(μ::FiniteMean)(n::Int) = μ.μ(getobs(μ.X, n))
length(μ::FiniteMean) = nobs(μ.X)
show(io::IO, μ::FiniteMean) = show(io, "FiniteMean($(μ.μ)")
unary_obswise(μ::FiniteMean, q::AVM) = unary_obswise(μ.μ, getobs(μ.X, q))

struct EmpiricalKernel{T<:LazyPDMat} <: Kernel
    Σ::T
end
@inline (k::EmpiricalKernel)(q::Int, q′::Int) = k.Σ[q, q′]

"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel <: Kernel
    k::Kernel
    X::AVM
end
FiniteKernel(Σ::LazyPDMat) = FiniteKernel(EmpiricalKernel(Σ), 1:size(Σ, 1))
(k::FiniteKernel)(n::Int, n′::Int) = k.k(getobs(k.X, n), getobs(k.X, n′))
size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? nobs(k.X) : 1
show(io::IO, k::FiniteKernel) = show(io, "FiniteKernel($(k.k))")
function binary_obswise(k::FiniteKernel, q::AVM, q′::AVM)
    return binary_obswise(k.k, getobs(k.X, q), getobs(k.X, q′))
end
function pairwise(k::FiniteKernel, q::AVM)
    return pairwise(k.k, getobs(k.X, q))
end
pairwise(k::FiniteKernel, q::AVM, q′::AVM) = pairwise(k.k, getobs(k.X, q), getobs(k.X, q′))


"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. Useful for defining
cross-covariance between a Finite kernel and other non-Finite kernels.
"""
struct LhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AVM
end
(k::LhsFiniteCrossKernel)(n::Int, x) = k.k(getobs(k.X, n), x)
size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? nobs(k.X) : size(k.k, N)
show(io::IO, k::LhsFiniteCrossKernel) = show(io, "LhsFiniteCrossKernel($(k.k))")
eachindex(k::LhsFiniteCrossKernel, N::Int) = 1:size(k, N)
function binary_obswise(k::LhsFiniteCrossKernel, q::AVM, X′::AVM)
    return binary_obswise(k.k, getobs(k.X, q), X′)
end
function pairwise(k::LhsFiniteCrossKernel, q::AVM, X′::AVM)
    return pairwise(k.k, getobs(k.X, q), X′)
end

"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X′::AVM
end
(k::RhsFiniteCrossKernel)(x, n′::Int) = k.k(x, getobs(k.X′, n′))
size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? nobs(k.X′) : size(k.k, N)
show(io::IO, k::RhsFiniteCrossKernel) = show(io, "RhsFiniteCrossKernel($(k.k))")
function binary_obswise(k::RhsFiniteCrossKernel, X::AVM, q′::AVM)
    return binary_obswise(k.k, X, getobs(k.X′, q′))
end
function pairwise(k::RhsFiniteCrossKernel, X::AVM, q′::AVM)
    return pairwise(k.k, X, getobs(k.X′, q′))
end

"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
additional data.
"""
struct FiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AVM
    X′::AVM
end
(k::FiniteCrossKernel)(n::Int, n′::Int) = k.k(getobs(k.X, n), getobs(k.X′, n′))
size(k::FiniteCrossKernel, N::Int) = N == 1 ? nobs(k.X) : (N == 2 ? nobs(k.X′) : 1)
show(io::IO, k::FiniteCrossKernel) = show(io, "FiniteCrossKernel($(k.k))")
function binary_obswise(k::FiniteCrossKernel, q::AVM, q′::AVM)
    return binary_obswise(k.k, getobs(k.X, q), getobs(k.X′, q′))
end
function pairwise(k::FiniteCrossKernel, q::AVM, q′::AVM)
    return pairwise(k.k, getobs(k.X, q), getobs(k.X′, q′))
end
