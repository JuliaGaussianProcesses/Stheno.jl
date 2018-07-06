import Base: eachindex, AbstractVector, AbstractMatrix, map
export FiniteMean, FiniteKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    FiniteCrossKernel

const IntVec = AbstractVector{<:Integer}

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean <: MeanFunction
    μ::MeanFunction
    X::AbstractVector
end
(μ::FiniteMean)(n) = μ.μ(getindex(μ.X, n))
finite(μ::MeanFunction, X::AbstractVector) = FiniteMean(μ, X)
eachindex(μ::FiniteMean) = eachindex(μ.X)
length(μ::FiniteMean) = length(μ.X)
show(io::IO, μ::FiniteMean) = show(io, "FiniteMean($(μ.μ)")
_map(μ::FiniteMean, q::IntVec) = map(μ.μ, μ.X[q])

# Sugar
map(μ::FiniteMean, ::Colon) = map(μ, eachindex(μ))

"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel <: Kernel
    k::Kernel
    X::AbstractVector
end
finite(k::Kernel, X::AbstractVector) = FiniteKernel(k, X)
(k::FiniteKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X, n′))
(k::FiniteKernel)(n) = k.k(k.X[n], k.X[n])
eachindex(k::FiniteKernel) = eachindex(k.X)
size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? length(k.X) : 1
show(io::IO, k::FiniteKernel) = show(io, "FiniteKernel($(k.k))")
_map(k::FiniteKernel, q::IntVec) = map(k.k, k.X[q])
_map(k::FiniteKernel, q::IntVec, q′::IntVec) = map(k.k, k.X[q], k.X[q′])
function _pairwise(k::FiniteKernel, q::IntVec)
    return pairwise(k.k, k.X[q])
end
_pairwise(k::FiniteKernel, q::IntVec, q′::IntVec) = pairwise(k.k, k.X[q], k.X[q′])

# Sugar
map(k::FiniteKernel, ::Colon) = map(k, eachindex(k))
map(k::FiniteKernel, ::Colon, ::Colon) = map(k, eachindex(k), eachindex(k))
pairwise(k::FiniteKernel, ::Colon) = pairwise(k, eachindex(k))
pairwise(k::FiniteKernel, ::Colon, ::Colon) = pairwise(k, eachindex(k), eachindex(k))
pairwise(k::FiniteKernel, ::Colon, q′::IntVec) = pairwise(k, eachindex(k), q′)
pairwise(k::FiniteKernel, q::IntVec, ::Colon) = eachindex(k, q, eachindex(k))

"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. Useful for defining
cross-covariance between a Finite kernel and other non-Finite kernels.
"""
struct LhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AbstractVector
end
(k::LhsFiniteCrossKernel)(n, x) = k.k(getindex(k.X, n), x)
lhsfinite(k::CrossKernel, X::AbstractVector) = LhsFiniteCrossKernel(k, X)
size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : size(k.k, N)
show(io::IO, k::LhsFiniteCrossKernel) = show(io, "LhsFiniteCrossKernel($(k.k))")
eachindex(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.k, 2)
_map(k::LhsFiniteCrossKernel, q::IntVec, X′::AV) = map(k.k, k.X[q], X′)
_pairwise(k::LhsFiniteCrossKernel, q::IntVec, X′::AV) = pairwise(k.k, k.X[q], X′)

# Sugar
map(k::LhsFiniteCrossKernel, ::Colon, X′::AV) = map(k, eachindex(k, 1), X′)
pairwise(k::LhsFiniteCrossKernel, ::Colon, X′::AV) = pairwise(k, eachindex(k, 1), X′)

"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X′::AbstractVector
end
(k::RhsFiniteCrossKernel)(x, n′) = k.k(x, getindex(k.X′, n′))
rhsfinite(k::CrossKernel, X′::AbstractVector) = RhsFiniteCrossKernel(k, X′)
size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? length(k.X′) : size(k.k, N)
show(io::IO, k::RhsFiniteCrossKernel) = show(io, "RhsFiniteCrossKernel($(k.k))")
eachindex(k::RhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.k, 1) : eachindex(k.X′)
_map(k::RhsFiniteCrossKernel, X::AV, q′::IntVec) = map(k.k, X, k.X′[q′])
_pairwise(k::RhsFiniteCrossKernel, X::AV, q′::IntVec) = pairwise(k.k, X, k.X′[q′])

# Sugar
map(k::RhsFiniteCrossKernel, X::AV, ::Colon) = map(k, X, eachindex(k, 2))
pairwise(k::RhsFiniteCrossKernel, X::AV, ::Colon) = pairwise(k, X, eachindex(k, 2))

"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
additional data.
"""
struct FiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AbstractVector
    X′::AbstractVector
end
(k::FiniteCrossKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X′, n′))
finite(k::CrossKernel, X::AV, X′::AV) = FiniteCrossKernel(k, X, X′)
size(k::FiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : (N == 2 ? length(k.X′) : 1)
show(io::IO, k::FiniteCrossKernel) = show(io, "FiniteCrossKernel($(k.k))")
eachindex(k::FiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.X′)
_map(k::FiniteCrossKernel, q::IntVec, q′::IntVec) = map(k.k, k.X[q], k.X′[q′])
_pairwise(k::FiniteCrossKernel, q::IntVec, q′::IntVec) = pairwise(k.k, k.X[q], k.X′[q′])

# Sugar
map(k::FiniteCrossKernel, ::Colon, ::Colon) = map(k, eachindex(k, 1), eachindex(k, 2))
function pairwise(k::FiniteCrossKernel, ::Colon, ::Colon)
    return pairwise(k, eachindex(k, 1), eachindex(k, 2))
end
pairwise(k::FiniteCrossKernel, ::Colon, X′::IntVec) = pairwise(k, eachindex(k, 1), X′)
pairwise(k::FiniteCrossKernel, X::IntVec, ::Colon) = pairwise(k, X, eachindex(k, 2))
