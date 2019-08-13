import Base: *

*(f, g::AbstractGP) = CompositeGP((*, f, g), g.gpc)
*(f::AbstractGP, g) = CompositeGP((*, g, f), f.gpc)
*(f::AbstractGP, g::AbstractGP) = ArgumentError("Cannot multiply two GPs together.")

const prod_args{Tf} = Tuple{typeof(*), Tf, <:AbstractGP}

mean_vector((_, σ, g)::prod_args, x::AV) = σ.(x) .* mean_vector(g, x)
mean_vector((_, σ, g)::prod_args{<:Real}, x::AV) = σ .* mean_vector(g, x)

function cov_mat((_, σ, g)::prod_args, x::AV)
    σx = σ.(x)
    return σx .* cov_mat(g, x) .* σx'
end
cov_mat((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov_mat(g, x)

function cov_mat((_, σ, g)::prod_args, x::AV, x′::AV)
    return σ.(x) .* cov_mat(g, x, x′) .* σ.(x′)'
end
cov_mat((_, σ, g)::prod_args{<:Real}, x::AV, x′::AV) = (σ^2) .* cov_mat(g, x, x′)

cov_mat_diag((_, σ, g)::prod_args, x::AV) = σ.(x).^2 .* cov_mat_diag(g, x)
cov_mat_diag((_, σ, g)::prod_args{<:Real}, x::AV) = (σ^2) .* cov_mat_diag(g, x)

xcov_mat((_, σ, f)::prod_args, f′::AGP, x::AV, x′::AV) = σ.(x) .* xcov_mat(f, f′, x, x′)
xcov_mat((_, σ, f)::prod_args{<:Real}, f′::AGP, x::AV, x′::AV) = σ .* xcov_mat(f, f′, x, x′)
xcov_mat(f::AGP, (_, σ, f′)::prod_args, x::AV, x′::AV) = xcov_mat(f, f′, x, x′) .* σ.(x′)
xcov_mat(f::AGP, (_, σ, f′)::prod_args{<:Real}, x::AV, x′::AV) = xcov_mat(f, f′, x, x′) .* σ
