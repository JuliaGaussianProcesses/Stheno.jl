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

# function cov((_, σ, g)::prod_args, x::AV, x′::AV)
#     return σ.(x) .* cov(g, x, x′) .* σ.(x′)'
# end

cov_diag((_, σ, g)::prod_args, x::AV) = σ.(x).^2 .* cov_diag(g, x)

cov((_, σ, f)::prod_args, f′::AbstractGP, x::AV, x′::AV) = σ.(x) .* cov(f, f′, x, x′)
cov(f::AbstractGP, (_, σ, f′)::prod_args, x::AV, x′::AV) = cov(f, f′, x, x′) .* (σ.(x′))'

cov_diag((_, σ, f)::prod_args, f′::AbstractGP, x::AV) = σ.(x) .* cov_diag(f, f′, x)
cov_diag(f::AbstractGP, (_, σ, f′)::prod_args, x::AV) = cov_diag(f, f′, x) .* σ.(x)

#
# Scale by a constant
#

mean_vector((_, σ, g)::prod_args{<:Real}, x::AV) = σ .* mean_vector(g, x)

cov((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov(g, x)
# cov((_, σ, g)::prod_args{<:Real}, x::AV, x′::AV) = (σ^2) .* cov(g, x, x′)
cov_diag((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov_diag(g, x)

cov((_, σ, f)::prod_args{<:Real}, f′::AbstractGP, x::AV, x′::AV) = σ .* cov(f, f′, x, x′)
cov(f::AbstractGP, (_, σ, f′)::prod_args{<:Real}, x::AV, x′::AV) = cov(f, f′, x, x′) .* σ

cov_diag((_, σ, f)::prod_args{<:Real}, f′::AbstractGP, x::AV) = σ .* cov_diag(f, f′, x)
cov_diag(f::AbstractGP, (_, σ, f′)::prod_args{<:Real}, x::AV) = cov_diag(f, f′, x) .* σ

# Use multiplication to define the negation of a GP
-(f::AbstractGP) = (-1) * f
