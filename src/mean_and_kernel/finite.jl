import Base: eachindex, AbstractVector, AbstractMatrix, map
export FiniteMean, FiniteKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel,
    FiniteCrossKernel

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean <: MeanFunction
    μ::MeanFunction
    X::ADS
end
(μ::FiniteMean)(n) = μ.μ(getindex(μ.X, n))
eachindex(μ::FiniteMean) = eachindex(μ.X)
length(μ::FiniteMean) = length(μ.X)
show(io::IO, μ::FiniteMean) = show(io, "FiniteMean($(μ.μ)")
map(μ::FiniteMean, q::AV{<:Integer}) = map(μ.μ, μ.X[q])

"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel <: Kernel
    k::Kernel
    X::ADS
end
FiniteKernel(Σ::LazyPDMat) = FiniteKernel(EmpiricalKernel(Σ), 1:size(Σ, 1))
(k::FiniteKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X, n′))
(k::FiniteKernel)(n) = k.k(k.X[n], k.X[n])
eachindex(k::FiniteKernel) = eachindex(k.X)
size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? length(k.X) : 1
show(io::IO, k::FiniteKernel) = show(io, "FiniteKernel($(k.k))")
map(k::FiniteKernel, q::DataSet, q′::DataSet) = map(k.k, k.X[q], k.X[q′])
function pairwise(k::FiniteKernel, q::DataSet)
    @show k.X, k.X[q], typeof(k.X[q]), typeof(k.X), typeof(q)
    return pairwise(k.k, k.X[q])
end
pairwise(k::FiniteKernel, q::DataSet, q′::DataSet) = pairwise(k.k, k.X[q], k.X[q′])

"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. Useful for defining
cross-covariance between a Finite kernel and other non-Finite kernels.
"""
struct LhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::ADS
end
(k::LhsFiniteCrossKernel)(n, x) = k.k(getindex(k.X, n), x)
size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : size(k.k, N)
show(io::IO, k::LhsFiniteCrossKernel) = show(io, "LhsFiniteCrossKernel($(k.k))")
eachindex(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.k, 2)
map(k::LhsFiniteCrossKernel, q::DataSet, X′::DataSet) = map(k.k, k.X[q], X′)
pairwise(k::LhsFiniteCrossKernel, q::DataSet, X′::DataSet) = pairwise(k.k, k.X[q], X′)

"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X′::ADS
end
(k::RhsFiniteCrossKernel)(x, n′) = k.k(x, getindex(k.X′, n′))
size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? length(k.X′) : size(k.k, N)
show(io::IO, k::RhsFiniteCrossKernel) = show(io, "RhsFiniteCrossKernel($(k.k))")
eachindex(k::RhsFiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.k, 1) : eachindex(k.X′)
map(k::RhsFiniteCrossKernel, X::DataSet, q′::DataSet) = map(k.k, X, k.X′[q′])
pairwise(k::RhsFiniteCrossKernel, X::DataSet, q′::DataSet) = pairwise(k.k, X, k.X′[q′])

"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
additional data.
"""
struct FiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::ADS
    X′::ADS
end
(k::FiniteCrossKernel)(n, n′) = k.k(getindex(k.X, n), getindex(k.X′, n′))
size(k::FiniteCrossKernel, N::Int) = N == 1 ? length(k.X) : (N == 2 ? length(k.X′) : 1)
show(io::IO, k::FiniteCrossKernel) = show(io, "FiniteCrossKernel($(k.k))")
eachindex(k::FiniteCrossKernel, N::Int) = N == 1 ? eachindex(k.X) : eachindex(k.X′)
map(k::FiniteCrossKernel, q::DataSet, q′::DataSet) = map(k.k, k.X[q], k.X′[q′])
pairwise(k::FiniteCrossKernel, q::DataSet, q′::DataSet) = pairwise(k.k, k.X[q], k.X′[q′])
