"""
    sample(rng::AbstractRNG, d::Union{GP, Vector}, N::Int=1)

Sample jointly from a single / multiple finite-dimensional GPs.
"""
function sample(rng::AbstractRNG, ds::Vector{GP}, N::Int)
    lin_sample = mean_vector(ds) .+ chol(cov(ds)).'randn(rng, sum(dims.(ds)), N)
    srt, fin = vcat(1, cumsum(dims.(ds))[1:end-1] .+ 1), cumsum(dims.(ds))
    return broadcast((srt, fin)->lin_sample[srt:fin, :], srt, fin)
end
sample(rng::AbstractRNG, ds::Vector{GP}) = reshape.(sample(rng, ds, 1), dims.(ds))
sample(rng::AbstractRNG, d::GP, N::Int) = sample(rng, Vector{GP}([d]), N)[1]
sample(rng::AbstractRNG, d::GP) = sample(rng, Vector{GP}([d]))[1]
