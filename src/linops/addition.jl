import Base: +, -

"""
    +(fa::AbstractGP, fb::AbstractGP)

Produces an AbstractGP `f` satisfying `f(x) = fa(x) + fb(x)`.
"""
function +(fa::AbstractGP, fb::AbstractGP)
    @assert fa.gpc === fb.gpc
    return CompositeGP((+, fa, fb), fa.gpc)
end
-(fa::AbstractGP, fb::AbstractGP) = fa + (-fb)

#
# Add two GPs
#

const add_args = Tuple{typeof(+), AbstractGP, AbstractGP}

mean_vector((_, fa, fb)::add_args, x::AV) = mean_vector(fa, x) .+ mean_vector(fb, x)

function cov((_, fa, fb)::add_args, x::AV)
    return cov(fa, x) .+ cov(fb, x) .+ cov(fa, fb, x, x) .+ cov(fb, fa, x, x)
end
function cov_diag((_, fa, fb)::add_args, x::AV)
    return +(
        cov_diag(fa, x), cov_diag(fb, x),
        cov_diag(fa, fb, x, x), cov_diag(fb, fa, x, x),
    )
end

function cov((_, fa, fb)::add_args, x::AV, x′::AV)
    return cov(fa, x, x′) .+ cov(fb, x, x′) .+ cov(fa, fb, x, x′) .+ cov(fb, fa, x, x′)
end
function cov_diag((_, fa, fb)::add_args, x::AV, x′::AV)
    return +(
        cov_diag(fa, x, x′), cov_diag(fb, x, x′),
        cov_diag(fa, fb, x, x′), cov_diag(fb, fa, x, x′),
    )
end

function cov((_, fa, fb)::add_args, f′::AbstractGP, x::AV, x′::AV)
    return cov(fa, f′, x, x′) .+ cov(fb, f′, x, x′)
end
function cov(f::AbstractGP, (_, fa, fb)::add_args, x::AV, x′::AV)
    return cov(f, fa, x, x′) .+ cov(f, fb, x, x′)
end

function cov_diag((_, fa, fb)::add_args, f′::AbstractGP, x::AV, x′::AV)
    return cov_diag(fa, f′, x, x′) .+ cov_diag(fb, f′, x, x′)
end
function cov_diag(f::AbstractGP, (_, fa, fb)::add_args, x::AV, x′::AV)
    return cov_diag(f, fa, x, x′) .+ cov_diag(f, fb, x, x′)
end


#
# Add a constant or known function to a GP -- just shifts the mean
#

+(b, f::AbstractGP) = CompositeGP((+, b, f), f.gpc)
+(f::AbstractGP, b) = b + f
-(b::Real, f::AbstractGP) = b + (-f)
-(f::AbstractGP, b::Real) = f + (-b)

const add_known{T} = Tuple{typeof(+), T, AbstractGP}

mean_vector((_, b, f)::add_known, x::AV) = b.(x) .+ mean_vector(f, x)
mean_vector((_, b, f)::add_known{<:Real}, x::AV) = b .+ mean_vector(f, x)

cov((_, b, f)::add_known, x::AV) = cov(f, x)
cov_diag((_, b, f)::add_known, x::AV) = cov_diag(f, x)

cov((_, b, f)::add_known, x::AV, x′::AV) = cov(f, x, x′)
cov_diag((_, b, f)::add_known, x::AV, x′::AV) = cov_diag(f, x, x′)

cov((_, b, f)::add_known, f′::AbstractGP, x::AV, x′::AV) = cov(f, f′, x, x′)
cov(f::AbstractGP, (_, b, f′)::add_known, x::AV, x′::AV) = cov(f, f′, x, x′)

cov_diag((_, b, f)::add_known, f′::AbstractGP, x::AV, x′::AV) = cov_diag(f, f′, x, x′)
cov_diag(f::AbstractGP, (_, b, f′)::add_known, x::AV, x′::AV) = cov_diag(f, f′, x, x′)
