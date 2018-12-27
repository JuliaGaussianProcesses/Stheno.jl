import Base: eachindex, map, ==
export FiniteMean, FiniteKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    FiniteCrossKernel

const IntVec = AbstractVector{<:Integer}


"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean{Tμ<:MeanFunction, TX<:AbstractVector} <: MeanFunction
    μ::Tμ
    X::TX
end
==(μ::FiniteMean, μ′::FiniteMean) = μ.μ == μ′.μ && μ.X == μ′.X
eachindex(μ::FiniteMean) = eachindex(μ.X)
length(μ::FiniteMean) = length(μ.X)

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
==(k::FiniteKernel, k′::FiniteKernel) = k.k == k′.k && k.X == k′.X
eachindex(k::FiniteKernel) = eachindex(k.X)
length(k::FiniteKernel) = length(k.X)

(k::FiniteKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X, n′))
map(k::FiniteKernel, ::Colon, ::Colon) = map(k.k, k.X, k.X)
pairwise(k::FiniteKernel, ::Colon, ::Colon) = pairwise(k.k, k.X, k.X)

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

==(k::LhsFiniteCrossKernel, k′::LhsFiniteCrossKernel) = k.k == k′.k && k.X == k′.X
size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : size(k.k, N)
eachindex(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.k, 2)

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

==(k::RhsFiniteCrossKernel, k′::RhsFiniteCrossKernel) = k.k == k′.k && k.X′ == k′.X′
size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? length(k.X′) : size(k.k, N)
eachindex(k::RhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.k, 1) : eachindex(k.X′)

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
==(k::FiniteCrossKernel, k′::FiniteCrossKernel) = k.k == k′.k && k.X == k′.X && k.X′ == k′.X′
size(k::FiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : (N == 2 ? length(k.X′) : 1)
eachindex(k::FiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.X′)

(k::FiniteCrossKernel)(n::Integer, n′::Integer) = k.k(k.X[n], k.X′[n′])
map(k::FiniteCrossKernel, ::Colon, ::Colon) = map(k.k, k.X, k.X′)
pairwise(k::FiniteCrossKernel, ::Colon, ::Colon) = pairwise(k.k, k.X, k.X′)



################################## Optimisations for zeros #################################

struct FiniteZeroMean{TX} <: MeanFunction
    X::TX
end
map(μ::FiniteZeroMean, ::Colon) = Zeros(length(μ.X))

struct FiniteZeroKernel{TX} <: Kernel
    X::TX
end
map(k::FiniteZeroKernel, ::Colon, ::Colon) = Zeros(length(k.X))
pairwise(k::FiniteZeroKernel, ::Colon, ::Colon) = Zeros(length(k.X), length(k.X))
map(k::FiniteZeroKernel, ::Colon) = map(k, :, :)
pairwise(k::FiniteZeroKernel, ::Colon) = pairwise(k, :, :)

struct FiniteZeroCrossKernel{TX, TX′} <: CrossKernel
    X::TX
    X′::TX′
end
pairwise(k::FiniteZeroCrossKernel, ::Colon, ::Colon) = Zeros(length(k.X), length(k.X′))

struct LhsFiniteZeroCrossKernel{TX} <: CrossKernel
    X::TX
end
function map(k::LhsFiniteZeroCrossKernel, ::Colon, X′::AV)
    @assert length(k.X) == length(X′)
    return Zeros(length(k.X))
end
pairwise(k::LhsFiniteZeroCrossKernel, ::Colon, X′::AV) = Zeros(length(k.X), length(X′))

struct RhsFiniteZeroCrossKernel{TX′} <: CrossKernel
    X′::TX′
end
function map(k::RhsFiniteZeroCrossKernel, X::AV, ::Colon)
    @assert length(X) == length(k.X′)
    return Zeros(length(X))
end
pairwise(k::RhsFiniteZeroCrossKernel, X::AV, ::Colon) = Zeros(length(X), length(k.X′))



######################################## Sugar #############################################

finite(μ::MeanFunction, X::AbstractVector) = FiniteMean(μ, X)
finite(μ::ZeroMean, X::AbstractVector) = FiniteZeroMean(X)
finite(μ::FiniteZeroMean, q::AbstractVector) = FiniteZeroMean(q)

finite(k::Kernel, X::AbstractVector) = FiniteKernel(k, X)
finite(k::ZeroKernel, X::AbstractVector) = FiniteZeroKernel(X)
finite(k::FiniteZeroKernel, q::AbstractVector) = FiniteZeroKernel(q)

finite(k::CrossKernel, X::AV, X′::AV) = FiniteCrossKernel(k, X, X′)
function finite(k::ZeroKernel, X::AV, X′::AV)
    return length(X) == length(X′) ? FiniteZeroKernel(X) : FiniteZeroCrossKernel(X, X′)
end
finite(k::FiniteZeroCrossKernel, q::AV, q′::AV) = FiniteZeroCrossKernel(q, q′)

const LhsFinite = Union{LhsFiniteCrossKernel, LhsFiniteZeroCrossKernel}
const RhsFinite = Union{RhsFiniteCrossKernel, RhsFiniteZeroCrossKernel}

lhsfinite(k::CrossKernel, X::AbstractVector) = LhsFiniteCrossKernel(k, X)
lhsfinite(k::ZeroKernel, X::AbstractVector) = LhsFiniteZeroCrossKernel(X)
lhsfinite(k::RhsFiniteCrossKernel, X::AbstractVector) = finite(k.k, X, k.X′)
lhsfinite(k::RhsFiniteZeroCrossKernel, X::AV) = FiniteZeroCrossKernel(X, k.X′)

rhsfinite(k::CrossKernel, X′::AbstractVector) = RhsFiniteCrossKernel(k, X′)
rhsfinite(k::ZeroKernel, X′::AbstractVector) = RhsFiniteZeroCrossKernel(X′)
rhsfinite(k::LhsFiniteCrossKernel, X′::AbstractVector) = finite(k.k, k.X, X′)
rhsfinite(k::LhsFiniteZeroCrossKernel, X′::AV) = FiniteZeroCrossKernel(k.X, X′)
