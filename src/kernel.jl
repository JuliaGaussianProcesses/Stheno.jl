import Base: +, *, ==
export KernelType, Kernel, EQ, RQ, Linear

"""
Determines whether a kernel is stationary or not and enables dispatch on this.
"""
abstract type KernelType end

""" Indicates that a `Kernel` is non-stationary. """
abstract type NonStationary <: KernelType end

""" Indicates that a `Kernel` is stationary. """
abstract type Stationary <: KernelType end

"""
    Kernel{T<:KernelType}

Supertype for all kernels; `T` indicates whether is `Stationary` or `NonStationary`.
"""
abstract type Kernel{T<:KernelType} end

"""
    Constant{T<:Real} <: Kernel{Stationary}

A rank 1 constant `Kernel`. Useful for consistency when creating composite Kernels,
but (almost certainly) shouldn't be used as a base `Kernel`.
"""
struct Constant{T<:Real} <: Kernel{Stationary}
    value::T
end
(k::Constant)(::Any, ::Any) = k.value
==(a::Constant, b::Constant) = a.value == b.value

"""
    EQ <: Kernel{Stationary}

The standardised Exponentiated Quadratic kernel with no free parameters.
"""
struct EQ <: Kernel{Stationary} end
(::EQ)(x::Real, y::Real) = exp(-0.5 * abs2(x - y))
==(::EQ, ::EQ) = true

"""
    RQ{T<:Real} <: Kernel{Stationary}

The standardised Rational Quadratic. `RQ(α)` creates an `RQ` `Kernel{Stationary}` whose
kurtosis is `α`.
"""
struct RQ{T<:Real} <: Kernel{Stationary}
    α::T
end
(k::RQ)(x::Real, y::Real) = (1 + 0.5 * abs2(x - y) / k.α)^(-k.α)
==(a::RQ, b::RQ) = a.α == b.α

"""
    Linear{T<:Real} <: Kernel{NonStationary}

Standardised linear kernel. `Linear(c)` creates a `Linear` `Kernel{NonStationary}` whose
intercept is `c`.
"""
struct Linear{T<:Real} <: Kernel{NonStationary}
    c::T
end
(k::Linear)(x::Real, y::Real) = (x - k.c) * (y - k.c)
==(a::Linear, b::Linear) = a.c == b.c

"""
    Composite{V, O, T<:Tuple{Kernel, N} where N} <: Kernel{V}

A `Composite` kernel is generated through the addition or multiplication of two existing
`Kernel` objects, or the addition or multiplication of an existing `Kernel` object and a
`Real`.
"""
struct Composite{V, O<:Function, T<:Tuple{Kernel, N} where N} <: Kernel{V}
    args::T
end

for op in (:+, :*)
    @eval begin
        $op(a::Real, b::Kernel) = $op(Constant(a), b)
        $op(a::Kernel, b::Real) = $op(a, Constant(b))
        $op(a::Ta, b::Tb) where {Ta<:Kernel, Tb<:Kernel} =
            Composite{NonStationary, typeof($op), Tuple{Ta, Tb}}((a, b))
        $op(a::Ta, b::Tb) where {Ta<:Kernel{Stationary}, Tb<:Kernel{Stationary}} =
            Composite{Stationary, typeof($op), Tuple{Ta, Tb}}((a, b))
        (k::Composite{<:KernelType, typeof($op)})(x::T, y::T) where T =
            $op(k.args[1](x, y), k.args[2](x, y))
        function ==(
            a::Composite{<:KernelType, typeof($op), <:Tuple{Kernel, N} where N},
            b::Composite{<:KernelType, typeof($op), <:Tuple{Kernel, N} where N},
        )
            return (a.args[1] == b.args[1] && a.args[2] == b.args[2]) ||
                   (a.args[2] == b.args[1] && a.args[1] == b.args[2])
        end
    end
end
