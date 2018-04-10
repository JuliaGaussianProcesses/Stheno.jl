export FiniteMean, FiniteKernel, FiniteCrossKernel

cov(v::Union{<:MeanFunction, <:CrossKernel}) = error("Attempted to compute ∞ object.")

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
mean(μ::FiniteMean, X::AM) = error("Attempted index into FiniteMean.")
size(μ::FiniteMean) = (size(μ.X, 1),)
size(μ::FiniteMean, n::Int) = n == 1 ? size(μ.X, 1) : 1
length(μ::FiniteMean) = size(μ.X, 1)
isfinite(μ::FiniteMean) = true
show(io::IO, μ::FiniteMean) = show(io, "FiniteMean($(μ.μ)")

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
xcov(k::FiniteKernel) = Matrix(cov(k))
cov(k::FiniteKernel) = cov(k.k, k.X)
cov(k::FiniteKernel, X::AM) = error("Attempted index into FiniteKernel.")
size(k::FiniteKernel) = (size(k.X, 1), size(k.X, 1))
size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? size(k.X, 1) : 1
isfinite(k::FiniteKernel) = true
isstationary(k::FiniteKernel) = false
show(io::IO, k::FiniteKernel) = show(io, "FiniteKernel($(k.k))")

"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct LhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AM
    LhsFiniteCrossKernel(k::CrossKernel, X::AM) = new(k, X)
    LhsFiniteCrossKernel(k::LhsFiniteCrossKernel, X::AM) =
        throw(error("Can't nest LhsFiniteCrossKernels"))
end
xcov(k::LhsFiniteCrossKernel, X′::AM) = xcov(k.k, k.X, X′)
==(k::T, k′::T) where T<:LhsFiniteCrossKernel = (k.k == k′.k) && (k.X == k′.X)

"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X′::AM
end
xcov(k::RhsFiniteCrossKernel, X::AM) = xcov(k.k, X, k.X′)
==(k::T, k′::T) where T<:RhsFiniteCrossKernel = (k.k == k′.k) && (k.X′ == k′.X′)

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
FiniteCrossKernel(k::LhsFiniteCrossKernel, X′::AM) = FiniteCrossKernel(k.k, k.X, X′)
FiniteCrossKernel(k::RhsFiniteCrossKernel, X::AM) = FiniteCrossKernel(k.k, X, k.X′)
(k::FiniteCrossKernel)(p::Int, q::Int) = k.k(view(k.X, p, :), view(k.X′, q, :))
xcov(k::FiniteCrossKernel) = xcov(k.k, k.X, k.X′)
xcov(k::FiniteCrossKernel, X::AM, X′::AM) = error("Attempted index into FiniteCrossKernel.")
size(k::FiniteCrossKernel) = (size(k.X, 1), size(k.X′, 1))
size(k::FiniteCrossKernel, N::Int) = N == 1 ? size(k.X, 1) : (N == 2 ? size(k.X′, 1) : 1)
isfinite(k::FiniteCrossKernel) = true
isstationary(k::FiniteCrossKernel) = false
show(io::IO, k::FiniteCrossKernel) = show(io, "FiniteCrossKernel($(k.k))")
