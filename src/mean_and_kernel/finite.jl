export FiniteMean, FiniteKernel, FiniteCrossKernel

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean <: MeanFunction
    μ::MeanFunction
    X::AbstractMatrix
end
(μ::FiniteMean)(p::Int) = μ.μ(view(μ.X, p, :))
mean(μ::FiniteMean) = mean(μ.μ, μ.X)
size(μ::FiniteMean) = (size(μ.X, 1),)
size(μ::FiniteMean, n::Int) = n == 1 ? size(μ.X, 1) : 1
isfinite(μ::FiniteMean) = true

"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel <: Kernel
    k::Kernel
    X::AM
end
(k::FiniteKernel)(p::Int, q::Int) = k.k(view(k.X, p, :), view(k.X, q, :))
cov(k::FiniteKernel) = cov(k.k, k.X)
size(k::FiniteKernel) = (size(k.X, 1), size(k.X, 1))
size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? size(k.X, 1) : 1
isfinite(k::FiniteKernel) = true
isstationary(k::FiniteKernel) = false


"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
additional data.
"""
struct FiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AM
    X′::AM
end
(k::FiniteCrossKernel)(p::Int, q::Int) = k.k(view(k.X, p, :), view(k.X′, q, :))
xcov(k::FiniteCrossKernel) = xcov(k, X, X′)
size(k::FiniteCrossKernel) = (size(k.X, 1), size(k.X′, 1))
size(k::FiniteCrossKernel, N::Int) = N == 1 ? size(k.X, 1) : (N == 2 ? size(k.X′, 1) : 1)
isfinite(k::FiniteCrossKernel) = true
isstationary(k::FiniteCrossKernel) = false

"""
    cov(k::Matrix{<:FiniteKernel})

Compute the covariance matrix for each of a set of finite kernels. This functionality might
(will) move elsewhere at some point.
"""
function cov(k::Matrix{<:FiniteKernel})
    rs_, cs_ = size.(k[:, 1], 1), size.(k[1, :], 2)
    rs = Vector{Int}(undef, length(rs_) + 1)
    cs = Vector{Int}(undef, length(cs_) + 1)
    cumsum!(view(rs, 2:length(rs_) + 1), rs_)
    cumsum!(view(cs, 2:length(cs_) + 1), cs_)
    rs[1], cs[1] = 0, 0
    K = Matrix{Float64}(undef, rs[end], cs[end])
    for I in CartesianIndices(k)
        K[rs[I[1]]+1:rs[I[1]+1], cs[I[2]]+1:cs[I[2]+1]] = cov(k[I[1], I[2]])
    end
    return K
end
