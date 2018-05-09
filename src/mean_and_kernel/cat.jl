export CatMean, CatKernel, CatCrossKernel

@inline nobs(X::AV{<:AVM}) = sum(nobs, X)
function getobs(X::AV{<:AVM}, n::Int)
    p = 1
    while n > nobs(X[p])
        n -= nobs(X[p])
        p += 1
    end
    return (p, getobs(X[p], n))
end

"""
    CatMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct CatMean <: MeanFunction
    μ::Vector
end
CatMean(μs::Vararg{<:MeanFunction}) = CatMean([μs...])
(μ::CatMean)(x::Tuple{Int, <:Any}) = μ.μ[x[1]](x[2])
length(μ::CatMean) = sum(length.(μ.μ))
==(μ::CatMean, μ′::CatMean) = μ.μ == μ′.μ
unary_obswise(μ::CatMean, X::AV{<:AVM}) = BlockVector(unary_obswise.(μ.μ, X))
unary_obswise(μ::CatMean, X::AVM) = unary_obswise(μ, [X])

"""
    CatCrossKernel <: CrossKernel

A cross kernel comprising lots of other kernels.
"""
struct CatCrossKernel <: CrossKernel
    ks::Matrix
end
CatCrossKernel(ks::Vector) = CatCrossKernel(reshape(ks, length(ks), 1))
CatCrossKernel(ks::RowVector) = CatCrossKernel(reshape(ks, 1, length(ks)))
size(k::CatCrossKernel, N::Int) = N == 1 ?
    sum(size.(k.ks[:, 1], Ref(1))) :
    N == 2 ? sum(size.(k.ks[1, :], Ref(2))) : 1
(k::CatCrossKernel)(x::Tuple{Int, <:Any}, x′::Tuple{Int, <:Any}) =
    k.ks[x[1], x′[1]](x[2], x′[2])
function pairwise(k::CatCrossKernel, X::AV{<:AVM}, X′::AV{<:AVM})
    Ω = BlockMatrix{Float64}(uninitialized_blocks, nobs.(X), nobs.(X′))
    for q in 1:nblocks(Ω, 2), p in 1:nblocks(Ω, 1)
        setblock!(Ω, pairwise(k.ks[p, q], X[p], X′[q]), p, q)
    end
    return Ω
end

"""
    CatKernel <: Kernel

A kernel comprising lots of other kernels. This is represented as a matrix whose diagonal
elements are `Kernels`, and whose off-diagonal elements are `CrossKernel`s. In the absence
of determining at either either compile- or construction-time whether or not this actually
constites a valid Mercer kernel, we take the construction of this type to be a promise on
the part of the caller that the thing they are constructing does indeed constitute a valid
Mercer kernel.

`ks_diag` represents the kernels on the diagonal of this matrix-valued kernel, and `ks_off`
represents the elements in the rest of the matrix. Only the upper-triangle will actually
be used.
"""
struct CatKernel <: Kernel
    ks_diag::Vector{<:Kernel}
    ks_off::Matrix{<:CrossKernel}
end
function (k::CatKernel)(x::Tuple{Int, <:Any}, x′::Tuple{Int, <:Any})
    if x[1] == x′[1]
        return k.ks_diag[x[1]](x[2], x′[2])
    elseif x[1] < x′[1]
        return k.ks_off[x[1], x′[1]](x[2], x′[2])
    else
        return k.ks_off[x′[1], x[1]](x′[2], x[2])'
    end
end
size(k::CatKernel, N::Int) = (N ∈ (1, 2)) ? sum(size.(k.ks_diag, 1)) : 1

binary_obswise(k::CatKernel, X::AV{<:AVM}) = BlockVector(binary_obswise.(k.ks_diag, X))
binary_obswise(k::CatKernel, X::AV{<:AVM}, X′::AV{<:AVM}) =
    BlockVector(binary_obswise.(k.ks_diag, X, X′))

function pairwise(k::CatKernel, X::AV{<:AVM})
    Σ = BlockMatrix{Float64}(uninitialized_blocks, nobs.(X), nobs.(X))
    for q in eachindex(k.ks_diag)
        setblock!(Σ, pairwise(k.ks_diag[q], X[q]), q, q)
        for p in 1:q-1
            setblock!(Σ, pairwise(k.ks_off[p, q], X[p], X[q]), p, q)
            setblock!(Σ, getblock(Σ, p, q)', q, p)
        end
    end
    return SquareDiagonal(Σ)
end
pairwise(k::CatKernel, X::AVM) = pairwise(k, [X])
function pairwise(k::CatKernel, X::AV{<:AVM}, X′::AV{<:AVM})
    Ω = BlockMatrix{Float64}(uninitialized_blocks, nobs.(X), nobs.(X′))
    for q in eachindex(k.ks_diag), p in eachindex(k.ks_diag)
        if p == q
            setblock!(Ω, pairwise(k.ks_diag[p], X[p], X′[p]), p, p)
        elseif p < q
            setblock!(Ω, pairwise(k.ks_off[p, q], X[p], X′[q]), p, q)
        else
            setblock!(Ω, pairwise(k.ks_off[q, p], X[p], X′[q]), p, q)
        end
    end
    return Ω
end

xcov(k::Union{<:CatCrossKernel, <:CatKernel}, X::AVM, X′::AVM) = xcov(k, [X], [X′])
xcov(k::Union{<:CatCrossKernel, <:CatKernel}, X::AV{<:AVM}, X′::AVM) = xcov(k, X, [X′])
xcov(k::Union{<:CatCrossKernel, <:CatKernel}, X::AVM, X′::AV{<:AVM}) = xcov(k, [X], X′)
