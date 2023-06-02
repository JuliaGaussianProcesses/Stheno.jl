import Base: +, -

"""
    +(fa::AbstractGP, fb::AbstractGP)

Produces an AbstractGP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
function +(fa::AbstractGP, fb::AbstractGP)
    @assert fa.gpc === fb.gpc
    return DerivedGP((+, fa, fb), fa.gpc)
end
-(fa::AbstractGP, fb::AbstractGP) = fa + (-fb)



#
# Add two GPs
#

mean(::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV) = mean(fa, x) .+ mean(fb, x)

function cov(::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV)
    return cov(fa, x) .+ cov(fb, x) .+ cov(fa, fb, x, x) .+ cov(fb, fa, x, x)
end
function var(::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV)
    return var(fa, x) .+ var(fb, x) .+ var(fa, fb, x, x) .+ var(fb, fa, x, x)
end

function cov(::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV, x′::AV)
    return cov(fa, x, x′) .+ cov(fb, x, x′) .+ cov(fa, fb, x, x′) .+ cov(fb, fa, x, x′)
end
function var(::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV, x′::AV)
    return var(fa, x, x′) .+ var(fb, x, x′) .+ var(fa, fb, x, x′) .+ var(fb, fa, x, x′)
end

function cov(::typeof(+), fa::AbstractGP, fb::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    return cov(fa, f′, x, x′) .+ cov(fb, f′, x, x′)
end
function cov(f::AbstractGP, ::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV, x′::AV)
    return cov(f, fa, x, x′) .+ cov(f, fb, x, x′)
end

function var(::typeof(+), fa::AbstractGP, fb::AbstractGP, f′::AbstractGP, x::AV, x′::AV)
    return var(fa, f′, x, x′) .+ var(fb, f′, x, x′)
end
function var(f::AbstractGP, ::typeof(+), fa::AbstractGP, fb::AbstractGP, x::AV, x′::AV)
    return var(f, fa, x, x′) .+ var(f, fb, x, x′)
end



#
# Add a constant or known function to an AbstractGP -- just shifts the mean
#

+(b, f::AbstractGP) = DerivedGP((+, b, f), f.gpc)
+(f::AbstractGP, b) = b + f
-(b::Real, f::AbstractGP) = b + (-f)
-(f::AbstractGP, b::Real) = f + (-b)

mean(::typeof(+), b, f::AbstractGP, x::AV) = b.(x) .+ mean(f, x)
mean(::typeof(+), b::Real, f::AbstractGP, x::AV) = b .+ mean(f, x)

cov(::typeof(+), b, f::AbstractGP, x::AV) = cov(f, x)
var(::typeof(+), b, f::AbstractGP, x::AV) = var(f, x)

cov(::typeof(+), b, f::AbstractGP, x::AV, x′::AV) = cov(f, x, x′)
var(::typeof(+), b, f::AbstractGP, x::AV, x′::AV) = var(f, x, x′)

cov(::typeof(+), b, f::AbstractGP, f′::AbstractGP, x::AV, x′::AV) = cov(f, f′, x, x′)
cov(f::AbstractGP, ::typeof(+), b, f′::AbstractGP, x::AV, x′::AV) = cov(f, f′, x, x′)

var(::typeof(+), b, f::AbstractGP, f′::AbstractGP, x::AV, x′::AV) = var(f, f′, x, x′)
var(f::AbstractGP, ::typeof(+), b, f′::AbstractGP, x::AV, x′::AV) = var(f, f′, x, x′)
