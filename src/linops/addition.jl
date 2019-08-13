import Base: +, -

"""
    +(fa::AbstractGP, fb::AbstractGP)

Produces an AbstractGP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
function +(fa::AbstractGP, fb::AbstractGP)
    @assert fa.gpc === fb.gpc
    return CompositeGP((+, fa, fb), fa.gpc)
end
-(fa::AbstractGP, fb::AbstractGP) = fa + (-b)

#
# Add two GPs
#

const add_args = Tuple{typeof(+), AbstractGP, AbstractGP}

mean_vector((_, fa, fb)::add_args, x::AV) = mean_vector(fa, x) .+ mean_vector(fb, x)

function cov((_, fa, fb)::add_args, x::AV)
    return cov(fa, x) .+ cov(fb, x) .+ xcov(fa, fb, x) .+ xcov(fb, fa, x)
end

function cov((_, fa, fb)::add_args, x::AV, x′::AV)
    return cov(fa, x, x′) .+ cov(fb, x, x′) .+ xcov(fa, fb, x, x′) .+ xcov(fb, fa, x, x′)
end

function cov_diag((_, fa, fb)::add_args, x::AV)
    return cov_diag(fa, x) .+ cov_diag(fb, x) .+ cov_diag(fa, fb, x) .+ cov_diag(fb, fa, x)
end

function xcov((_, fa, fb)::add_args, f′::AbstractGP, x::AV, x′::AV)
    return xcov(fa, f′, x, x′) .+ xcov(fb, f′, x, x′)
end
function xcov(f::AbstractGP, (_, fa, fb)::add_args, x::AV, x′::AV)
    return xcov(f, fa, x, x′) .+ xcov(f, fb, x, x′)
end

function xcov_diag((_, fa, fb)::add_args, f′::AbstractGP, x::AV)
    return xcov_diag(fa, f′, x) .+ xcov_diag(fb, f′, x)
end
function xcov_diag(f::AbstractGP, (_, fa, fb)::add_args, x::AV)
    return xcov_diag(f, fa, x) .+ xcov_diag(f, fb, x)
end

#
# Add a constant or known function to a GP -- just shifts the mean
#

+(b, f::AbstractGP) = CompositeGP((+, b, f), f.gpc)
+(f::AbstractGP, b) = b + f
-(b::Real, f::AbstractGP) = b + (-f)
-(f::AbstractGP, b::Real) = f + (-b)

const add_known{T} = Tuple{T, AbstractGP}

mean_vector((_, b, f)::add_known, x::AV) = b.(x) .+ mean_vector(f, x)

cov((_, b, f)::add_known, x::AV) = cov(f, x)
cov((_, b, f)::add_known, x::AV, x′::AV) = cov(f, x, x′)
cov_diag((_, b, f)::add_known, x::AV) = cov_diag(f, x)

xcov((_, b, f)::add_known, f′::AbstractGP, x::AV, x′::AV) = xcov(f, f′, x, x′)
xcov(f::AbstractGP, (_, b, f′)::add_known, x::AV, x′::AV) = xcov(f, f′, x, x′)

xcov_diag((_, b, f)::add_known, f′::AbstractGP, x::AV) = xcov_diag(f, f′, x)
xcov_diag(f::AbstractGP, (_, b, f′)::add_known, x::AV) = xcov_diag(f, f′, x)
