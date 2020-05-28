const valid_kernel_type = Scaled{<:AV{<:Real}, <:Stretched{<:AV{<:Real}, <:EQ}}

derivative(f::GP{<:ZeroMean, <:valid_kernel_type}) = CompositeGP((derivative, f,), f.gpc)

const derivative_args = Tuple{typeof(derivative), <:GP{<:ZeroMean, <:valid_kernel_type}}

mean_vector((_, f)::derivative_args, x::AV{<:Real}) = zero(x)

function cov((_, f)::derivative_args, x::AV{<:Real})
    k_stretch = f.k.k
    λ = first(k_stretch.f.(k_stretch.a))^2
    return λ .* cov(f, x) .* (1 .- λ .* (x .- x').^2)
end

function cov((_, f)::derivative_args, x::AV{<:Real}, x′::AV{<:Real})
    k_stretch = f.k.k
    λ = first(k_stretch.f.(k_stretch.a))^2
    return λ .* cov(f, x, x′) .* (1 .- λ .* (x .- x′').^2)
end

function cov((_, f)::derivative_args, f′::GP, x::AV{<:Real}, x′::AV{<:Real})
    @assert f === f′ # Require that the other process is in fact the one we're targetting.
    k_stretch = f.k.k
    λ = first(k_stretch.f.(k_stretch.a))^2
    return -λ .* (x .- x′') .* cov(f, x, x′)
end

function cov(f′, (_, f)::derivative_args, x::AV{<:Real}, x′::AV{<:Real})
    @assert f === f′ # Require that the other process is in fact the one we're targetting.
    k_stretch = f.k.k
    λ = first(k_stretch.f.(k_stretch.a))^2
    return λ .* (x .- x′') .* cov(f, x, x′)
end

# Dummy implementations to satisfy the interface.
cov_diag(df::derivative_args, x::AV{<:Real}) = diag(cov(df, x))
cov_diag(df::derivative_args, x::AV{<:Real}, x′::AV{<:Real}) = diag(cov(df, x, x′))
cov_diag(df::derivative_args, f::GP, x::AV{<:Real}, x′::AV{<:Real}) = diag(cov(df, f, x, x′))
cov_diag(f::GP, df::derivative_args, x::AV{<:Real}, x′::AV{<:Real}) = diag(cov(f, df, x, x′))
