"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean{Tμ<:MeanFunction, TX<:AbstractVector} <: MeanFunction
    μ::Tμ
    X::TX
end
map(μ::FiniteMean, ::Colon) = map(μ.μ, μ.X)


"""
    FiniteKernel <: Kernel

A kernel valued on a finite index set. Has a method of `cov` which requires no additional
data.
"""
struct FiniteKernel{Tk<:Kernel, TX<:AbstractVector} <: Kernel
    k::Tk
    X::TX
end

# Binary methods.
map(k::FiniteKernel, ::Colon, ::Colon) = map(k.k, k.X, k.X)
pairwise(k::FiniteKernel, ::Colon, ::Colon) = pairwise(k.k, k.X, k.X)

# Unary methods.
map(k::FiniteKernel, ::Colon) = map(k.k, k.X)
pairwise(k::FiniteKernel, ::Colon) = pairwise(k.k, k.X)


"""
    FiniteCrossKernel <: CrossKernel

A cross kernel valued on a finite index set. Has a method of `cov` which requires no
additional data.
"""
struct FiniteCrossKernel{Tk<:CrossKernel, TX<:AV, TX′<:AV} <: CrossKernel
    k::Tk
    X::TX
    X′::TX′
end
map(k::FiniteCrossKernel, ::Colon, ::Colon) = map(k.k, k.X, k.X′)
pairwise(k::FiniteCrossKernel, ::Colon, ::Colon) = pairwise(k.k, k.X, k.X′)


# """
#     LhsFiniteCrossKernel <: CrossKernel

# A cross kernel whose first argument is defined on a finite index set. Useful for defining
# cross-covariance between a Finite kernel and other non-Finite kernels.
# """
# struct LhsFiniteCrossKernel{Tk<:CrossKernel, TX<:AbstractVector} <: CrossKernel
#     k::Tk
#     X::TX
# end
# (k::LhsFiniteCrossKernel)(n, x) = k.k(k.X[n], x)
# map(k::LhsFiniteCrossKernel, ::Colon, X′::AV) = map(k.k, k.X, X′)
# pairwise(k::LhsFiniteCrossKernel, ::Colon, X′::AV) = pairwise(k.k, k.X, X′)


# """
#     RhsFiniteCrossKernel <: CrossKernel

# A cross kernel whose second argument is defined on a finite index set. You can't really do
# anything with this object other than use it to construct other objects.
# """
# struct RhsFiniteCrossKernel{Tk<:CrossKernel, TX′<:AbstractVector} <: CrossKernel
#     k::Tk
#     X′::TX′
# end
# (k::RhsFiniteCrossKernel)(x, n′) = k.k(x, k.X′[n′])
# map(k::RhsFiniteCrossKernel, X::AV, ::Colon) = map(k.k, X, k.X′)
# pairwise(k::RhsFiniteCrossKernel, X::AV, ::Colon) = pairwise(k.k, X, k.X′)
