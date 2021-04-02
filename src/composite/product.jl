import Base: *, -

"""
    *(f, g::AbstractGP)

Produce an `AbstractGP` `h` satisfying to `h(x) = f(x) * g(x)`, for some deterministic
function `f`.

If `f isa Real`, then `h(x) = f * g(x)`.
"""
*(f, g::AbstractGP) = CompositeGP((*, f, g), g.gpc)
*(f::AbstractGP, g) = CompositeGP((*, g, f), f.gpc)
*(f::AbstractGP, g::AbstractGP) = throw(ArgumentError("Cannot multiply two GPs together."))

const prod_args{Tf} = Tuple{typeof(*), Tf, <:AbstractGP}

#
# Scale by a function
#

mean((_, σ, g)::prod_args, x::AV) = σ.(x) .* mean(g, x)

function cov((_, σ, g)::prod_args, x::AV)
    σx = σ.(x)
    return σx .* cov(g, x) .* σx'
end
var((_, σ, g)::prod_args, x::AV) = σ.(x).^2 .* var(g, x)

function cov((_, σ, g)::prod_args, x::AV, x′::AV)
    return σ.(x) .* cov(g, x, x′) .* σ.(x′)'
end
function var((_, σ, g)::prod_args, x::AV, x′::AV)
    return σ.(x) .* var(g, x, x′) .* σ.(x′)
end

cov((_, σ, f)::prod_args, f′::AbstractGP, x::AV, x′::AV) = σ.(x) .* cov(f, f′, x, x′)
cov(f::AbstractGP, (_, σ, f′)::prod_args, x::AV, x′::AV) = cov(f, f′, x, x′) .* (σ.(x′))'

function var((_, σ, f)::prod_args, f′::AbstractGP, x::AV, x′::AV)
    return σ.(x) .* var(f, f′, x, x′)
end
function var(f::AbstractGP, (_, σ, f′)::prod_args, x::AV, x′::AV)
    return var(f, f′, x, x′) .* σ.(x′)
end

#
# Scale by a constant
#

mean((_, σ, g)::prod_args{<:Real}, x::AV) = σ .* mean(g, x)

cov((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov(g, x)
var((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* var(g, x)

cov((_, σ, g)::prod_args{<:Real}, x::AV, x′::AV) = (σ^2) .* cov(g, x, x′)
var((_, σ, g)::prod_args{<:Real}, x::AV, x′::AV) = (σ^2) .* var(g, x, x′)

cov((_, σ, f)::prod_args{<:Real}, f′::AbstractGP, x::AV, x′::AV) = σ .* cov(f, f′, x, x′)
cov(f::AbstractGP, (_, σ, f′)::prod_args{<:Real}, x::AV, x′::AV) = cov(f, f′, x, x′) .* σ

function var((_, σ, f)::prod_args{<:Real}, f′::AbstractGP, x::AV, x′::AV)
    return σ .* var(f, f′, x, x′)
end
function var(f::AbstractGP, (_, σ, f′)::prod_args{<:Real}, x::AV, x′::AV)
    return var(f, f′, x, x′) .* σ
end

# Use multiplication to define the negation of a GP
-(f::AbstractGP) = (-1) * f
