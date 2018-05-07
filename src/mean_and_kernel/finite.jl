import Base: eachindex
export FiniteMean, FiniteKernel, FiniteCrossKernel, LhsFiniteCrossKernel, RhsFiniteCrossKernel

"""
    FiniteMean <: Function

A mean function defined on a finite index set. Has a method of `mean` which requires no
additional data.
"""
struct FiniteMean{TX<:AMRV} <: MeanFunction
    μ::MeanFunction
    X::TX
end
# size(μ::FiniteMean, N::Int) = N == 1 ? size(μ.X, 2) : 1
length(μ::FiniteMean) = size(μ.X, 2)
show(io::IO, μ::FiniteMean) = show(io, "FiniteMean($(μ.μ)")
eachindex(μ::FiniteMean) = RowVector(1:length(μ))
mean(μ::FiniteMean) = unary_colwise(μ, eachindex(μ))
unary_colwise(μ::FiniteMean, r::AbstractMatrix{<:Integer}) =
    unary_colwise(μ, RowVector(reshape(r, length(r))))

unary_colwise(μ::FiniteMean{<:RowVector}, r::RowVector{<:Integer}) =
    unary_colwise(μ.μ, r == eachindex(μ) ? μ.X : μ.X[r'])
unary_colwise(μ::FiniteMean{<:AbstractMatrix}, r::RowVector{<:Integer}) =
    unary_colwise(μ.μ, r == eachindex(μ) ? μ.X : μ.X[:, r'])


# mean(μ::FiniteMean) = mean(μ, eachindex(μ))
# mean(μ::FiniteMean, r::AVM{<:Integer}) = mean(μ, reshape(r, length(r)))
# mean(μ::FiniteMean, r::AV{<:Integer}) = mean(μ.μ, r == eachindex(μ) ? μ.X : μ.X[r, :])

# """
#     FiniteKernel <: Kernel

# A kernel valued on a finite index set. Has a method of `cov` which requires no additional
# data.
# """
# struct FiniteKernel <: Kernel
#     k::Kernel
#     X::AVM
# end
# size(k::FiniteKernel, N::Int) = N ∈ (1, 2) ? size(k.X, 1) : 1
# show(io::IO, k::FiniteKernel) = show(io, "FiniteKernel($(k.k))")
# eachindex(k::FiniteKernel) = 1:size(k.X, 1)
# cov(k::FiniteKernel) = cov(k, eachindex(k))
# xcov(k::FiniteKernel) = Matrix(cov(k))
# marginal_cov(k::FiniteKernel) = marginal_cov(k, eachindex(k))
# cov(k::FiniteKernel, r::AV{<:Integer}) = cov(k.k, r == eachindex(k) ? k.X : k.X[r, :])
# xcov(k::FiniteKernel, r::AVM{<:Integer}, c::AVM{<:Integer}) =
#     xcov(k, reshape(r, length(r)), reshape(c, length(c)))
# function xcov(k::FiniteKernel, r::AV{<:Integer}, c::AV{<:Integer})
#     X = r == eachindex(k) ? k.X : k.X[r, :]
#     X′ = c == eachindex(k) ? k.X : k.X[c, :]
#     return xcov(k.k, X, X′)
# end
# marginal_cov(k::FiniteKernel, r::AV{<:Integer}) =
#     marginal_cov(k.k, r == eachindex(k) ? k.X : k.X[r, :])

# """
#     LhsFiniteCrossKernel <: CrossKernel

# A cross kernel whose first argument is defined on a finite index set. Useful for defining
# covariance between a Finite kernel and 
# """
# struct LhsFiniteCrossKernel <: CrossKernel
#     k::CrossKernel
#     X::AVM
# end
# size(k::LhsFiniteCrossKernel, N::Int) = N == 1 ? size(k.X, 1) : Inf
# show(io::IO, k::LhsFiniteCrossKernel) = show(io, "LhsFiniteCrossKernel($(k.k))")
# xcov(k::LhsFiniteCrossKernel, r::AVM{<:Integer}, X′::AVM) = xcov(k, reshape(r, length(r)), X′)
# function xcov(k::LhsFiniteCrossKernel, r::AV{<:Integer}, X′::AVM)
#     X = r == eachindex(k, 1) ? k.X : k.X[r, :]
#     return xcov(k.k, X, X′)
# end

# """
#     RhsFiniteCrossKernel <: CrossKernel

# A cross kernel whose second argument is defined on a finite index set. You can't really do
# anything with this object other than use it to construct other objects.
# """
# struct RhsFiniteCrossKernel <: CrossKernel
#     k::CrossKernel
#     X′::AVM
# end
# size(k::RhsFiniteCrossKernel, N::Int) = N == 2 ? size(k.X′, 1) : Inf
# show(io::IO, k::RhsFiniteCrossKernel) = show(io, "RhsFiniteCrossKernel($(k.k))")
# xcov(k::RhsFiniteCrossKernel, X::AVM, c::AVM{<:Integer}) = xcov(k, X, reshape(c, length(c)))
# function xcov(k::RhsFiniteCrossKernel, X::AVM, c::AV{<:Integer})
#     X′ = c == eachindex(k, 2) ? k.X′ : k.X′[c, :] 
#     return xcov(k.k, X, X′)
# end

# """
#     FiniteCrossKernel <: CrossKernel

# A cross kernel valued on a finite index set. Has a method of `xcov` which requires no
# additional data.
# """
# struct FiniteCrossKernel <: CrossKernel
#     k::CrossKernel
#     X::AVM
#     X′::AVM
# end
# size(k::FiniteCrossKernel, N::Int) = N == 1 ? size(k.X, 1) : (N == 2 ? size(k.X′, 1) : 1)
# show(io::IO, k::FiniteCrossKernel) = show(io, "FiniteCrossKernel($(k.k))")
# eachindex(k, N::Int) = N == 1 ? (1:size(k, 1)) : (1:size(k, 2))
# xcov(k::FiniteCrossKernel) = xcov(k, eachindex(k, 1), eachindex(k, 2))
# xcov(k::FiniteCrossKernel, r::AVM{<:Integer}, c::AVM{<:Integer}) =
#     xcov(k, reshape(r, length(r)), reshape(c, length(c)))
# function xcov(k::FiniteCrossKernel, r::AV{<:Integer}, c::AV{<:Integer})
#     X = r == eachindex(k, 1) ? k.X : k.X[r, :]
#     X′ = c == eachindex(k, 2) ? k.X′ : k.X′[c, :]
#     return xcov(k.k, X, X′)
# end
