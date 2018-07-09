"""
    AbstractVector(μ::MeanFunction)

Convert `μ` into an `AbstractVector` if such a representation exists.
"""
function AbstractVector(μ::MeanFunction)
    @assert isfinite(length(μ))
    return map(μ, eachindex(μ))
end
AbstractVector(μ::EmpiricalMean) = μ.μ

"""
    AbstractMatrix(k::Kernel)

Convert `k` into an `AbstractMatrix`, if such a representation exists.
"""
function AbstractMatrix(k::Kernel)
    @assert isfinite(size(k, 1))
    return pairwise(k, eachindex(k, 1))
end
AbstractMatrix(k::EmpiricalKernel) = k.Σ

"""
    AbstractMatrix(k::CrossKernel)

Convert `k` into an `AbstractMatrix`, if such a representation exists.
"""
function AbstractMatrix(k::CrossKernel)
    @assert isfinite(size(k, 1))
    @assert isfinite(size(k, 2))
    return pairwise(k, eachindex(k, 1), eachindex(k, 2))
end
