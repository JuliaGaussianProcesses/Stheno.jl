export FiniteMean, FiniteKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    FiniteCrossKernel

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean{Tμ<:MeanFunction, TX<:AbstractVector} <: MeanFunction
    μ::Tμ
    X::TX
end
(μ::FiniteMean)(n) = μ.μ(getindex(μ.X, n))
map(μ::FiniteMean, ::Colon) = map(μ.μ, μ.X)


"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel{Tk<:Kernel, TX<:AbstractVector} <: Kernel
    k::Tk
    X::TX
end

# Binary methods.
(k::FiniteKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X, n′))
map(k::FiniteKernel, ::Colon, ::Colon) = map(k.k, k.X, k.X)
pairwise(k::FiniteKernel, ::Colon, ::Colon) = pairwise(k.k, k.X, k.X)

# Unary methods.
(k::FiniteKernel)(n) = k.k(k.X[n], k.X[n])
map(k::FiniteKernel, ::Colon) = map(k.k, k.X)
pairwise(k::FiniteKernel, ::Colon) = pairwise(k.k, k.X)


"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. Useful for defining
cross-covariance between a Finite kernel and other non-Finite kernels.
"""
struct LhsFiniteCrossKernel{Tk<:CrossKernel, TX<:AbstractVector} <: CrossKernel
    k::Tk
    X::TX
end
(k::LhsFiniteCrossKernel)(n, x) = k.k(k.X[n], x)
map(k::LhsFiniteCrossKernel, ::Colon, X′::AV) = map(k.k, k.X, X′)
pairwise(k::LhsFiniteCrossKernel, ::Colon, X′::AV) = pairwise(k.k, k.X, X′)


"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel{Tk<:CrossKernel, TX′<:AbstractVector} <: CrossKernel
    k::Tk
    X′::TX′
end
(k::RhsFiniteCrossKernel)(x, n′) = k.k(x, k.X′[n′])
map(k::RhsFiniteCrossKernel, X::AV, ::Colon) = map(k.k, X, k.X′)
pairwise(k::RhsFiniteCrossKernel, X::AV, ::Colon) = pairwise(k.k, X, k.X′)


"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
additional data.
"""
struct FiniteCrossKernel{Tk<:CrossKernel, TX<:AV, TX′<:AV} <: CrossKernel
    k::Tk
    X::TX
    X′::TX′
end
(k::FiniteCrossKernel)(n::Integer, n′::Integer) = k.k(k.X[n], k.X′[n′])
map(k::FiniteCrossKernel, ::Colon, ::Colon) = map(k.k, k.X, k.X′)
pairwise(k::FiniteCrossKernel, ::Colon, ::Colon) = pairwise(k.k, k.X, k.X′)



######################################## Sugar #############################################

finite(μ::MeanFunction, X::AbstractVector) = FiniteMean(μ, X)
finite(μ::ZeroMean, X::AbstractVector) = FiniteZeroMean(X)

finite(k::Kernel, X::AbstractVector) = FiniteKernel(k, X)
finite(k::ZeroKernel, X::AbstractVector) = FiniteZeroKernel(X)

finite(k::CrossKernel, X::AV, X′::AV) = FiniteCrossKernel(k, X, X′)
function finite(k::ZeroKernel, X::AV, X′::AV)
    return length(X) == length(X′) ? FiniteZeroKernel(X) : FiniteZeroCrossKernel(X, X′)
end

const LhsFinite = Union{LhsFiniteCrossKernel}
const RhsFinite = Union{RhsFiniteCrossKernel}

lhsfinite(k::CrossKernel, X::AbstractVector) = LhsFiniteCrossKernel(k, X)
lhsfinite(k::ZeroKernel, X::AbstractVector) = LhsFiniteZeroCrossKernel(X)
lhsfinite(k::RhsFiniteCrossKernel, X::AbstractVector) = finite(k.k, X, k.X′)

rhsfinite(k::CrossKernel, X′::AbstractVector) = RhsFiniteCrossKernel(k, X′)
rhsfinite(k::ZeroKernel, X′::AbstractVector) = RhsFiniteZeroCrossKernel(X′)
rhsfinite(k::LhsFiniteCrossKernel, X′::AbstractVector) = finite(k.k, k.X, X′)
