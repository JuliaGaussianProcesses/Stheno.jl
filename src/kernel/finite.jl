# Covariance functions for FiniteGPs / Multivariate Normals.
export Finite, LeftFinite, RightFinite

"""
    Finite <: Kernel{NonStationary}

A kernel on a finite index set.
"""
struct Finite{T<:ColOrRowVec, V<:ColOrRowVec, Tk<:Any} <: Kernel{NonStationary}
    k::Tk
    x::T
    y::V
end
Finite(k, x) = Finite(k, x, x)
@inline (k::Finite)(p::Int, q::Int) = k.k(k.x[p], k.y[q])
==(a::Finite, b::Finite) = a.k == b.k && a.x == b.x && a.y == b.y
function Base.show(io::IO, k::Finite)
    print(io, "Finite of size $(length(k.x))x$(length(k.y)) with base kernel $(k.k)")
end

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
function Base.show(io::IO, k::LeftFinite)
    print(io, "LeftFinite of size $(length(k.x)) with base kernel $(k.k)")
end

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
function Base.show(io::IO, k::RightFinite)
    print(io, "RightFinite of size $(length(k.y)) with base kernel $(k.k)")
end
