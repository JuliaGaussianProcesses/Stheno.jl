import Base: eachindex
export FiniteMean, FiniteKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    FiniteCrossKernel

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean <: MeanFunction
    μ::MeanFunction
    X::AVM
end
function (μ::FiniteMean)(n::Int)
    return μ.μ(getobs(μ.X, n))
end
length(μ::FiniteMean) = nobs(μ.X)
show(io::IO, μ::FiniteMean) = show(io, "FiniteMean($(μ.μ)")
eachindex(μ::FiniteMean) = 1:length(μ)
mean(μ::FiniteMean) = unary_obswise(μ, eachindex(μ))

"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel <: Kernel
    k::Kernel
    X::AVM
end
(k::FiniteKernel)(n::Int, n′::Int) = k.k(getobs(k.X, n), getobs(k.X, n′))
size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? nobs(k.X) : 1
show(io::IO, k::FiniteKernel) = show(io, "FiniteKernel($(k.k))")
eachindex(k::FiniteKernel) = 1:nobs(k.X)
cov(k::FiniteKernel) = pairwise(k, eachindex(k))
xcov(k::FiniteKernel) = pairwise(k, eachindex(k))

"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. Useful for defining
cross-covariance between a Finite kernel and other non-Finite kernels.
"""
struct LhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AVM
end
(k::LhsFiniteCrossKernel)(n::Int, x) = k.k(getobs(k.X, n), x)
size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? nobs(k.X) : Inf
show(io::IO, k::LhsFiniteCrossKernel) = show(io, "LhsFiniteCrossKernel($(k.k))")

"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X′::AVM
end
(k::RhsFiniteCrossKernel)(x, n′::Int) = k.k(x, getobs(k.X′, n′))
size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? nobs(k.X′) : Inf
show(io::IO, k::RhsFiniteCrossKernel) = show(io, "RhsFiniteCrossKernel($(k.k))")

"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
additional data.
"""
struct FiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AVM
    X′::AVM
end
(k::FiniteCrossKernel)(n::Int, n′::Int) = k.k(getobs(k.X, n), getobs(k.X′, n′))
size(k::FiniteCrossKernel, N::Int) = N == 1 ? nobs(k.X) : (N == 2 ? nobs(k.X′) : 1)
show(io::IO, k::FiniteCrossKernel) = show(io, "FiniteCrossKernel($(k.k))")
xcov(k::FiniteCrossKernel) = pairwise(k, eachindex(k, 1), eachindex(k, 2))

# General eachindex implementation.
eachindex(k::CrossKernel, N::Int) = 1:size(k, N)
