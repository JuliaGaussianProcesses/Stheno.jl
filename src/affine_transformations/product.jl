import Base: *, -

"""
    *(f, g::AbstractGP)

Produce an `AbstractGP` `h` satisfying to `h(x) = f(x) * g(x)`, for some deterministic
function `f`.

If `f isa Real`, then `h(x) = f * g(x)`.
"""
*(f, g::AbstractGP) = DerivedGP((*, f, g), g.gpc)
*(f::AbstractGP, g) = DerivedGP((*, g, f), f.gpc)
*(f::AbstractGP, g::AbstractGP) = throw(ArgumentError("Cannot multiply two GPs together."))

const prod_args{Tf} = Tuple{typeof(*),Tf,<:AbstractGP}

#
# Scale by a function
#


mean(::typeof(*), σ, g::AbstractGP, x::AV) = σ.(x) .* mean(g, x)

function cov(::typeof(*), σ, g::AbstractGP, x::AV)
    σx = σ.(x)
    return σx .* cov(g, x) .* σx'
end
var(::typeof(*), σ, g::AbstractGP, x::AV) = σ.(x) .^ 2 .* var(g, x)

function cov(::typeof(*), σ, g::AbstractGP, x::AV, x′::AV)
    return σ.(x) .* cov(g, x, x′) .* σ.(x′)'
end
function var(::typeof(*), σ, g::AbstractGP, x::AV, x′::AV)
    return σ.(x) .* var(g, x, x′) .* σ.(x′)
end

cov(::typeof(*), σ, f::AbstractGP, f′::AbstractGP, x::AV, x′::AV) = σ.(x) .* cov(f, f′, x, x′)
cov(f::AbstractGP, ::typeof(*), σ, f′::AbstractGP, x::AV, x′::AV) = cov(f, f′, x, x′) .* (σ.(x′))'

function var(::typeof(*), σ, f::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    return σ.(x) .* var(f, f′, x, x′)
end
function var(f::AbstractGP, ::typeof(*), σ, f′::AbstractGP, x::AV, x′::AV)
    return var(f, f′, x, x′) .* σ.(x′)
end

#
# Scale by a constant
#

mean(::typeof(*), σ::Real, g::AbstractGP, x::AV) = σ .* mean(g, x)

cov(::typeof(*), σ::Real, g::AbstractGP, x::AV) = (σ^2) .* cov(g, x)
var(::typeof(*), σ::Real, g::AbstractGP, x::AV) = (σ^2) .* var(g, x)

cov(::typeof(*), σ::Real, g::AbstractGP, x::AV, x′::AV) = (σ^2) .* cov(g, x, x′)
var(::typeof(*), σ::Real, g::AbstractGP, x::AV, x′::AV) = (σ^2) .* var(g, x, x′)

cov(::typeof(*), σ::Real, f::AbstractGP, f′::AbstractGP, x::AV, x′::AV) = σ .* cov(f, f′, x, x′)
cov(f::AbstractGP, ::typeof(*), σ::Real, f′::AbstractGP, x::AV, x′::AV) = cov(f, f′, x, x′) .* σ

function var(::typeof(*), σ::Real, f::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    return σ .* var(f, f′, x, x′)
end
function var(f::AbstractGP, ::typeof(*), σ::Real, f′::AbstractGP, x::AV, x′::AV)
    return var(f, f′, x, x′) .* σ
end

# Use multiplication to define the negation of a GP
-(f::AbstractGP) = (-1) * f
