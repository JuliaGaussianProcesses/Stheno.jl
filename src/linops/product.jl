import Base: *, -

*(f, g::AbstractGP) = CompositeGP((*, f, g), g.gpc)
*(f::AbstractGP, g) = CompositeGP((*, g, f), f.gpc)
*(f::AbstractGP, g::AbstractGP) = ArgumentError("Cannot multiply two GPs together.")

const prod_args{Tf} = Tuple{typeof(*), Tf, <:AbstractGP}

#
# Scale by a function
#

mean_vector((_, σ, g)::prod_args, x::AV) = σ.(x) .* mean_vector(g, x)

function cov((_, σ, g)::prod_args, x::AV)
    σx = σ.(x)
    return σx .* cov(g, x) .* σx'
end

function cov((_, σ, g)::prod_args, x::AV, x′::AV)
    return σ.(x) .* cov(g, x, x′) .* σ.(x′)'
end

cov_diag((_, σ, g)::prod_args, x::AV) = σ.(x).^2 .* cov_diag(g, x)

xcov((_, σ, f)::prod_args, f′::AbstractGP, x::AV, x′::AV) = σ.(x) .* xcov(f, f′, x, x′)
xcov(f::AbstractGP, (_, σ, f′)::prod_args, x::AV, x′::AV) = xcov(f, f′, x, x′) .* (σ.(x′))'

xcov_diag((_, σ, f)::prod_args, f′::AbstractGP, x::AV) = σ.(x) .* xcov_diag(f, f′, x)
xcov_diag(f::AbstractGP, (_, σ, f′)::prod_args, x::AV) = xcov_diag(f, f′, x) .* σ.(x)

function sample(rng::AbstractRNG, (_, σ, g)::prod_args, x::AV, S::Int)
    return σ.(x) .* sample(rng, g, x, S)
end

#
# Scale by a constant
#

mean_vector((_, σ, g)::prod_args{<:Real}, x::AV) = σ .* mean_vector(g, x)

cov((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov(g, x)
cov((_, σ, g)::prod_args{<:Real}, x::AV, x′::AV) = (σ^2) .* cov(g, x, x′)
cov_diag((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov_diag(g, x)

xcov((_, σ, f)::prod_args{<:Real}, f′::AbstractGP, x::AV, x′::AV) = σ .* xcov(f, f′, x, x′)
xcov(f::AbstractGP, (_, σ, f′)::prod_args{<:Real}, x::AV, x′::AV) = xcov(f, f′, x, x′) .* σ

xcov_diag((_, σ, f)::prod_args{<:Real}, f′::AbstractGP, x::AV) = σ .* xcov_diag(f, f′, x)
xcov_diag(f::AbstractGP, (_, σ, f′)::prod_args{<:Real}, x::AV) = xcov_diag(f, f′, x) .* σ

function sample(rng::AbstractRNG, (_, σ, g)::prod_args{<:Real}, x::AV, S::Int)
    return σ .* sample(rng, g, x, S)
end

# Use multiplication to define the negation of a GP
-(f::AbstractGP) = (-1) * f
