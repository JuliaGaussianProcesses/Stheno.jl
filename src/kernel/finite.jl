# Covariance functions for FiniteGPs / Multivariate Normals.
import Base: size, show
export Finite, LeftFinite, RightFinite

"""
    Finite <: Kernel{NonStationary}

A kernel on a finite index set.
"""
struct Finite{Tx<:ColOrRowVec, Ty<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    x::Tx
    y::Ty
end
Finite(k, x) = Finite(k, x, x)
@inline (k::Finite)(p::Int, q::Int) = k.k(k.x[p], k.y[q])
==(a::Finite, b::Finite) = a.k == b.k && a.x == b.x && a.y == b.y
show(io::IO, k::Finite) =
    print(io, "Finite of size $(length(k.x))x$(length(k.y)) with base kernel $(k.k)")
size(k::Finite) = (length(k.x), length(k.y))
size(k::Finite, n::Int) = n == 1 ? length(k.x) : (n == 2 ? length(k.y) : 1)
dims(k::Finite) = length(k.x) # This is a hack! Needs to me made more robust.
isfinite(::Finite) = true

"""
    LeftFinite <: Kernel{NonStationary}

A kernel who's first (left) argument is from a finite index set.
"""
struct LeftFinite{T<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    x::T
end
@inline (k::LeftFinite)(p::Int, q) = k.k(k.x[p], q)
==(a::LeftFinite, b::LeftFinite) = a.k == b.k && a.x == b.x
show(io::IO, k::LeftFinite) = print(io, "LeftFinite, size = $(length(k.x)), kernel = $(k.k)")
size(k::LeftFinite, n::Int) =
    n == 1 ?
        length(k.x) :
        error("size(k::LeftFinite, 2) undefined.")

"""
    RightFinite <: Kernel{NonStationary}

A kernel who's second (right) argument is from a finite index set.
"""
struct RightFinite{T<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    y::T
end
@inline (k::RightFinite)(p, q::Int) = k.k(p, k.y[q])
==(a::RightFinite, b::RightFinite) = a.k == b.k && a.y == b.y
show(io::IO, k::RightFinite) =
    print(io, "RightFinite of size $(length(k.y)) with base kernel $(k.k)")
size(k::RightFinite, n::Int) =
    n == 1 ?
        error("size(k::RightFinite, 1) undefined.") :
        length(k.y)
