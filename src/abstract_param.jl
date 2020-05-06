export AbstractParameter, Parameter, value


abstract type AbstractParameter{T} end
const AP{T} = AbstractParameter{T}

# Sometime when setup our models, we require it's parameters to follow
# certain constrains, e.g. the support for scaling factor `σ` & length scale `l`
# of a GP kernel should be ℜ⁺, however, adding those constrains make the optimization
# process less convenient. `Parameter` is used to handle this, it's bijector is used
# to map constrained variable to unconstrained one, which is used during the optimization,
# and the inverse of the bijector does the reverse, which will provide us with the constrained
# variables we need.
struct Parameter{T, N, fT<:Bijector{N}} <: AP{T}
    x::AbstractArray{T}
    f::fT
    
    function Parameter{T}(y, f::fT) where {T<:Real, fT}
        N = ndims(y)
        x = inv(f)(y)
        if N == 0
            return new{T, N, fT}(T[x], f)
        else
            return new{T, N, fT}(x, f)
        end
    end
end

Parameter(y, f) = Parameter{eltype(y)}(y, f)

# interface
value(y::Parameter{T, 0}) where {T} = y.f(first(y.x))
value(y::Parameter) = y.f(y.x)

# if no constrain is added, it will use `Identity` bijector by default
Parameter(y) = Parameter(y, Identity{ndims(y)}())

# if we want positivity constrain, then use `Exp` bijector
Parameter(y, ::Val{:pos}) = Parameter(y, Bijectors.Exp{ndims(y)}())

