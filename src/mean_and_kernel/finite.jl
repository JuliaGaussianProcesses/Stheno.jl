import Base: eachindex, AbstractVector, AbstractMatrix, map, ==
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
==(μ::FiniteMean, μ′::FiniteMean) = μ.μ == μ′.μ && μ.X == μ′.X
eachindex(μ::FiniteMean) = eachindex(μ.X)
length(μ::FiniteMean) = length(μ.X)
_map(μ::FiniteMean, q::IntVec) = map(μ.μ, μ.X[q])
function print(io::IO, μ::FiniteMean)
    println(io, "FiniteMean ($(length(μ))):")
    print(io, Shunted(Shunt(4, ' '), μ.μ))
end

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
(k::FiniteKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X, n′))
(k::FiniteKernel)(n) = k.k(k.X[n], k.X[n])
==(k::FiniteKernel, k′::FiniteKernel) = k.k == k′.k && k.X == k′.X
eachindex(k::FiniteKernel) = eachindex(k.X)
length(k::FiniteKernel) = length(k.X)
function print(io::IO, k::FiniteKernel)
    println(io, "FiniteKernel $(size(k)):")
    print(io, Shunted(Shunt(4, ' '), k.k))
end
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
==(k::LhsFiniteCrossKernel, k′::LhsFiniteCrossKernel) = k.k == k′.k && k.X == k′.X
size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : size(k.k, N)
eachindex(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.k, 2)
_map(k::LhsFiniteCrossKernel, q::IntVec, X′::AV) = map(k.k, k.X[q], X′)
_pairwise(k::LhsFiniteCrossKernel, q::IntVec, X′::AV) = pairwise(k.k, k.X[q], X′)
function print(io::IO, k::LhsFiniteCrossKernel)
    println(io, "LhsFiniteCrossKernel $(size(k)):")
    print(io, Shunted(Shunt(4, ' '), k.k))
end

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
==(k::RhsFiniteCrossKernel, k′::RhsFiniteCrossKernel) = k.k == k′.k && k.X′ == k′.X′
size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? length(k.X′) : size(k.k, N)
eachindex(k::RhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.k, 1) : eachindex(k.X′)
_map(k::RhsFiniteCrossKernel, X::AV, q′::IntVec) = map(k.k, X, k.X′[q′])
_pairwise(k::RhsFiniteCrossKernel, X::AV, q′::IntVec) = pairwise(k.k, X, k.X′[q′])
function print(io::IO, k::RhsFiniteCrossKernel)
    println(io, "RhsFiniteCrossKernel $(size(k)):")
    print(io, Shunted(Shunt(4, ' '), k.k))
end

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
==(k::FiniteCrossKernel, k′::FiniteCrossKernel) = k.k == k′.k && k.X == k′.X && k.X′ == k′.X′
size(k::FiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : (N == 2 ? length(k.X′) : 1)
eachindex(k::FiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.X′)
_map(k::FiniteCrossKernel, q::IntVec, q′::IntVec) = map(k.k, k.X[q], k.X′[q′])
_pairwise(k::FiniteCrossKernel, q::IntVec, q′::IntVec) = pairwise(k.k, k.X[q], k.X′[q′])
function print(io::IO, k::FiniteCrossKernel)
    println(io, "FiniteCrossKernel $(size(k)):")
    print(io, Shunted(Shunt(4, ' '), k.k))
end

function zero(k::CrossKernel)
    if size(k, 1) < Inf && size(k, 2) < Inf
        return FiniteZeroCrossKernel(eachindex(k, 1), eachindex(k, 2))
    elseif size(k, 1) < Inf
        return LhsFiniteZeroCrossKernel(eachindex(k, 1))
    elseif size(k, 2) < Inf
        return RhsFiniteZeroCrossKernel(eachindex(k, 2))
    else
        return ZeroKernel{Float64}()
    end
end

# Sugar
map(k::FiniteCrossKernel, ::Colon, ::Colon) = map(k, eachindex(k, 1), eachindex(k, 2))
function pairwise(k::FiniteCrossKernel, ::Colon, ::Colon)
    return pairwise(k, eachindex(k, 1), eachindex(k, 2))
end
pairwise(k::FiniteCrossKernel, ::Colon, X′::IntVec) = pairwise(k, eachindex(k, 1), X′)
pairwise(k::FiniteCrossKernel, X::IntVec, ::Colon) = pairwise(k, X, eachindex(k, 2))



################################## Optimisations for zeros #################################

struct FiniteZeroMean{TX} <: MeanFunction
    X::TX
end
length(μ::FiniteZeroMean) = length(μ.X)
==(μ::FiniteZeroMean, μ′::FiniteZeroMean) = length(μ) == length(μ′)
eachindex(μ::FiniteZeroMean) = eachindex(μ.X)
print(io::IO, μ::FiniteZeroMean) = print(io, "FiniteZeroMean ($(length(μ)))")
_map(μ::FiniteZeroMean, q::AV) = Zeros(length(q))

struct FiniteZeroKernel{TX} <: Kernel
    X::TX
end
length(k::FiniteZeroKernel) = length(k.X)
==(k::FiniteZeroKernel, k′::FiniteZeroKernel) = length(k) == length(k′)
eachindex(k::FiniteZeroKernel) = eachindex(k.X)
print(io::IO, k::FiniteZeroKernel) = print(io, "FiniteZeroKernel $(size(k))")
_map(k::FiniteZeroKernel, q::AV) = Zeros(length(q))
_map(k::FiniteZeroKernel, q::AV, q′::AV) = Zeros(length(q))
_pairwise(k::FiniteZeroKernel, q::AV) = Zeros(length(q), length(q))
_pairwise(k::FiniteZeroKernel, q::AV, q′::AV) = Zeros(length(q), length(q′))

+(x::FiniteZeroMean, x′::FiniteZeroMean) = zero(x)

struct FiniteZeroCrossKernel{TX, TX′} <: CrossKernel
    X::TX
    X′::TX′
end
size(k::FiniteZeroCrossKernel, dim::Int) = dim == 1 ? length(k.X) : length(k.X′)
==(k::FiniteZeroCrossKernel, k′::FiniteZeroCrossKernel) = size(k) == size(k′)
eachindex(k::FiniteZeroCrossKernel, dim::Int) = dim == 1 ? eachindex(k.X) : eachindex(k.X′)
print(io::IO, k::FiniteZeroCrossKernel) = print(io, "FiniteZeroCrossKernel $(size(k))")
_map(k::FiniteZeroCrossKernel, q::AV) = Zeros(length(q))
_map(k::FiniteZeroCrossKernel, q::AV, q′::AV) = Zeros(length(q))
_pairwise(k::FiniteZeroCrossKernel, q::AV) = Zeros(length(q), length(q))
_pairwise(k::FiniteZeroCrossKernel, q::AV, q′::AV) = Zeros(length(q), length(q′))

struct LhsFiniteZeroCrossKernel{TX} <: CrossKernel
    X::TX
end
size(k::LhsFiniteZeroCrossKernel, dim::Int) = dim == 1 ? length(k.X) : Inf
==(k::LhsFiniteZeroCrossKernel, k′::LhsFiniteZeroCrossKernel) = size(k) == size(k′)
function eachindex(k::LhsFiniteZeroCrossKernel, dim::Int)
    return dim == 1 ? eachindex(k.X) : eachindex(ZeroKernel{Float64}(), 2)
end
function print(io::IO, k::LhsFiniteZeroCrossKernel)
    print(io, "LhsFiniteZeroCrossKernel $(size(k))")
end
_map(k::LhsFiniteZeroCrossKernel, q::AV, X′::AV) = Zeros(length(q))
_pairwise(k::LhsFiniteZeroCrossKernel, q::AV, X′::AV) = Zeros(length(q), length(X′))

struct RhsFiniteZeroCrossKernel{TX′} <: CrossKernel
    X′::TX′
end
size(k::RhsFiniteZeroCrossKernel, dim::Int) = dim == 1 ? Inf : length(k.X′)
==(k::RhsFiniteZeroCrossKernel, k′::RhsFiniteZeroCrossKernel) = size(k) == size(k′)
function eachindex(k::RhsFiniteZeroCrossKernel, dim::Int)
    return dim == 1 ? eachindex(ZeroKernel{Float64}(), 1) : eachindex(k.X′)
end
function print(io::IO, k::RhsFiniteZeroCrossKernel)
    print(io, "RhsFiniteZeroCrossKernel $(size(k))")
end
_map(k::RhsFiniteZeroCrossKernel, X::AV, q′::AV) = Zeros(length(X))
_pairwise(k::RhsFiniteZeroCrossKernel, X::AV, q′::AV) = Zeros(length(X), length(q′))

# More sugar.
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
