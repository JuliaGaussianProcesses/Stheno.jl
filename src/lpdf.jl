"""
    lpdf(d::GP, f::AbstractVector{<:Real})

Returns the log probability density of `f` under `d`. Dims `d` must be finite.
"""
lpdf(d::GP, f::AbstractVector{<:Real}) =
    -0.5 * (dims(d) * log(2π) * logdet(cov(d)) + invquad(cov(d), f .- mean_vector(d)))

