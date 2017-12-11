"""
    lpdf(d::GP, f::AbstractVector{<:Real})

Returns the log probability density of `f` under `d`. Dims `d` must be finite.
"""
lpdf(d::GP, f::AbstractVector{<:Real}) =
    -0.5 * (dims(d) * log(2π) * logdet(cov(d)) + invquad(cov(d), f .- mean_vector(d)))

"""
    lpdf(a::Vector{Assignment}})

Returns the log probability density observing the assignments `a` jointly.
"""
function lpdf(a::Vector{Assignment})
    f, y = [c̄.f for c̄ in a], [c̄.y for c̄ in a]
    Σ = cov(f)
    δΣinvδ = invquad(Σ, vcat(y...) .- mean_vector(f))
    return -0.5 * (sum(dims.(f)) * log(2π) * logdet(Σ) + δΣinvδ)
end
lpdf(a::Assignment...) = lpdf([a...])
