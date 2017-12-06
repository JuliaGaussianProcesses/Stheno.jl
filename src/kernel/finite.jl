# Covariance functions for FiniteGPs / Multivariate Normals.
import Base: size, show
export Finite, LhsFinite, RhsFinite

"""
    LhsFinite <: Kernel{NonStationary}

A kernel who's first (left) argument is from a finite index set.
"""
struct LhsFinite{T<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    x::T
end
@inline (k::LhsFinite)(p::Int, q) = k.k(k.x[p], q)
==(a::LhsFinite, b::LhsFinite) = a.k == b.k && a.x == b.x
show(io::IO, k::LhsFinite) = print(io, "LhsFinite, size = $(length(k.x)), kernel = $(k.k)")
size(k::LhsFinite, n::Int) =
    n == 1 ?
        length(k.x) :
        error("size(k::LhsFinite, 2) undefined.")

"""
    RhsFinite <: Kernel{NonStationary}

A kernel who's second (right) argument is from a finite index set.
"""
struct RhsFinite{T<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    y::T
end
@inline (k::RhsFinite)(p, q::Int) = k.k(p, k.y[q])
==(a::RhsFinite, b::RhsFinite) = a.k == b.k && a.y == b.y
show(io::IO, k::RhsFinite) =
    print(io, "RhsFinite of size $(length(k.y)) with base kernel $(k.k)")
size(k::RhsFinite, n::Int) =
    n == 1 ?
        error("size(k::RhsFinite, 1) undefined.") :
        length(k.y)

"""
    Finite <: Kernel{NonStationary}

A kernel on a finite index set.
"""
struct Finite{Tx<:ColOrRowVec, Ty<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    x::Tx
    y::Ty
    Finite(k::Tk, x::Tx, y::Ty) where {Tk, Tx, Ty} = new{Tx, Ty, Tk}(k, x, y)
    Finite(k::Tk, x::Tx) where {Tk, Tx} = new{Tx, Tx, Tk}(k, x, x)
    Finite(k::LhsFinite{Tx, Tk}, y::Ty) where {Tk, Tx, Ty} = new{Tx, Ty, Tk}(k.k, k.x, y)
    Finite(k::RhsFinite{Ty, Tk}, x::Tx) where {Tk, Tx, Ty} = new{Tx, Ty, Tk}(k.k, x, k.y)
end
@inline (k::Finite)(p::Int, q::Int) = k.k(k.x[p], k.y[q])
==(a::Finite, b::Finite) = a.k == b.k && a.x == b.x && a.y == b.y
show(io::IO, k::Finite) =
    print(io, "Finite of size $(length(k.x))x$(length(k.y)) with base kernel $(k.k)")
size(k::Finite) = (length(k.x), length(k.y))
size(k::Finite, n::Int) = n == 1 ? length(k.x) : (n == 2 ? length(k.y) : 1)
dims(k::Finite) = length(k.x) # This is a hack! Needs to me made more robust.
isfinite(::Finite) = true
