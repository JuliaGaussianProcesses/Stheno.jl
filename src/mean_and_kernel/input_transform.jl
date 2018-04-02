export InputTransformedKernel, Index, Periodic, Transform, input_transform

"""
    InputTransformedKernel <: Kernel

An `InputTransformedKernel` kernel is one which applies a transformation `f` to its inputs
prior to applying a Kernel `k` that it owns.
"""
struct InputTransformedKernel{Tk<:Kernel, Tf} <: Kernel
    k::Tk
    f::Tf
end
const Transform = InputTransformedKernel
isstationary(::Type{<:Transform{Tk}}) where Tk<:Kernel = isstationary(Tk)

@inline (k::Transform)(x::Tin, y::Tin) where Tin = k.k(k.f(x), k.f(y))

==(k1::Transform{Tk, Tf}, k2::Transform{Tk, Tf}) where {Tk, Tf} =
    (k1.k == k2.k) && (k1.f == k2.f)
dims(k::Transform) = dims(k.k)
kernel(k::Transform) = k.k
input_transform(k::Transform) = k.f

"""
    Index{N}

A parametric singleton type used to indicate to which dimension of an input a particular
`Transform` should be applied. For example,
```
x = [5.0, 4.0]
kt = Transform(k, Index{2})
kt(x, x) == k(x[2], x[2])
```
"""
struct Index{N, Tf}
    f::Tf
    function Index{N}() where N
        f = x->x[N]
        return new{N, typeof(f)}(f)
    end
end
@inline (idx::Index)(x) = idx.f(x)

Index{N}(k::Kernel) where N = Transform(k, Index{N}())

"""
    Periodic <: Kernel

Make a periodic kernel by applying the transformation `T:θ→(cos(2πθ), sin(2πθ))` to inputs
and computing `k(T(θ)[1], T(θ′)[1]) * k(T(θ)[2], T(θ′)[2])`.
"""
struct Periodic{Tk} <: Kernel
    k::Tk
end
==(k1::Periodic{Tk}, k2::Periodic{Tk}) where Tk = k1.k == k2.k
@inline (k::Periodic)(θ::Real, θ′::Real) =
    k.k(cos(2π * θ), cos(2π * θ′)) * k.k(sin(2π * θ), sin(2π * θ′))

