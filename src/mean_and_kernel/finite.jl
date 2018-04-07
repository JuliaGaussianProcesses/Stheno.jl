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
length(μ::FiniteMean) = size(μ.X, 1)
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
xcov(k::FiniteCrossKernel) = xcov(k.k, k.X, k.X′)
size(k::FiniteCrossKernel) = (size(k.X, 1), size(k.X′, 1))
size(k::FiniteCrossKernel, N::Int) = N == 1 ? size(k.X, 1) : (N == 2 ? size(k.X′, 1) : 1)
isfinite(k::FiniteCrossKernel) = true
isstationary(k::FiniteCrossKernel) = false

"""
    LhsFiniteCrossKernel <: CrossKernel

A cross kernel whose first argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct LhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AM
end

"""
    RhsFiniteCrossKernel <: CrossKernel

A cross kernel whose second argument is defined on a finite index set. You can't really do
anything with this object other than use it to construct other objects.
"""
struct RhsFiniteCrossKernel <: CrossKernel
    k::CrossKernel
    X::AM
end

==(k::T, k′::T) where T<:Union{LhsFiniteCrossKernel, RhsFiniteCrossKernel} =
    (k.k == k′.k) && (k.X == k′.X)
