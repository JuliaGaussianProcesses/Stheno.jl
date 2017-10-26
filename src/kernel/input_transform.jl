export InputTransformedKernel, input_transform, Index, Periodic, Transform

"""
    InputTransformedKernel{T<:KernelType} <: Kernel{T}

An `InputTransformedKernel` kernel is one which applies a transformation `f` to its inputs
prior to applying a Kernel `k` that it owns.
"""
struct InputTransformedKernel{T<:KernelType, Tk<:Kernel{T}, Tf} <: Kernel{T}
    k::Tk
    f::Tf
    InputTransformedKernel(k::Tk, f::Tf) where {T<:KernelType, Tk<:Kernel{T}, Tf} =
        new{T, Tk, Tf}(k, f)
end
const Transform = InputTransformedKernel

@inline (k::InputTransformedKernel)(x::Tin, y::Tin) where Tin =
    kernel(k)(input_transform(k)(x), input_transform(k)(y))

==(
    k1::InputTransformedKernel{T, Tk, Tf},
    k2::InputTransformedKernel{T, Tk, Tf},
) where {T, Tk, Tf} =
    kernel(k1) == kernel(k2)

"""
    kernel(k::InputTransformedKernel)

Get the kernel to whos inputs `k` applies an input domain transform.
"""
kernel(k::InputTransformedKernel) = k.k

"""
    input_transform(k::InputTransformedKernel)

Get the input domain transform which `k` applies to `kernel(k)`.
"""
input_transform(k::InputTransformedKernel) = k.f

"""
    Index{N}

A parametric singleton type used to indicate to which dimension of an input a particular
`InputTransformedKernel` should be applied. For example,
```
x = [5.0, 4.0]
kt = InputTransformedKernel(k, Index{2})
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

function Index{N}(k::Kernel) where N
    return InputTransformedKernel(k, Index{N}())
end

"""
    Periodic <: Kernel

Make a periodic kernel by applying the transformation `T:θ→(cos(2πθ), sin(2πθ))` to inputs
and computing `k(T(θ)[1], T(θ′)[1]) * k(T(θ)[2], T(θ′)[2])`.
"""
struct Periodic{T<:KernelType, Tk} <: Kernel{T}
    k::Tk
    Periodic(k::Kernel{T}) where T<:KernelType = new{T, typeof(k)}(k)
end
==(k1::Periodic{T, Tk}, k2::Periodic{T, Tk}) where {T, Tk} = k1.k == k2.k
@inline (k::Periodic)(θ::Real, θ′::Real) =
    k.k(cos(2π * θ), cos(2π * θ′)) * k.k(sin(2π * θ), sin(2π * θ′))
