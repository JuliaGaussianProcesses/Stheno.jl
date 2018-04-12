export CatMean, CatKernel, CatCrossKernel

"""
    CatMean <: MeanFunction

(Vertically) concatentated mean functions.
"""
struct CatMean <: MeanFunction
    μ::Vector
end
CatMean(μs::Vararg{<:MeanFunction}) = CatMean([μs...])
length(μ::CatMean) = sum(length.(μ.μ))
mean(μ::CatMean, X::AV{<:AVM}) = BlockVector(mean.(μ.μ, X))
mean(μ::CatMean, X::AVM) = mean(μ, [X])
==(μ::CatMean, μ′::CatMean) = μ.μ == μ′.μ

"""
    CatCrossKernel <: CrossKernel

A cross kernel comprising lots of other kernels. 

Doesn't currently implement a three- argument method for `xcov` since, in the absence of
`BlockArray` usage, there is no obvious way to elegantly determine uniquely which rows of
`X` and `X′` should be passed to which kernel. This may be implemented in the future when
`BlockArrays` is used as an integral part of Stheno, but until then we simply for the user
to only create finite `CatCrossKernel`s.
"""
struct CatCrossKernel <: CrossKernel
    ks::Matrix
end
CatCrossKernel(ks::Vector) = CatCrossKernel(reshape(ks, length(ks), 1))
CatCrossKernel(ks::RowVector) = CatCrossKernel(reshape(ks, 1, length(ks)))
size(k::CatCrossKernel) = (size(k, 1), size(k, 2))
size(k::CatCrossKernel, N::Int) = N == 1 ?
    sum(size.(k.ks[:, 1], Ref(1))) :
    N == 2 ? sum(size.(k.ks[1, :], Ref(2))) : 1
function xcov(k::CatCrossKernel, X::AV{<:AVM}, X′::AV{<:AVM})
    Ω = BlockMatrix{Float64}(uninitialized_blocks, size.(X, 1), size.(X′, 1))
    for q in 1:nblocks(Ω, 2), p in 1:nblocks(Ω, 1)
        setblock!(Ω, xcov(k.ks[p, q], X[p], X′[q]), p, q)
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
size(k::CatKernel, N::Int) = (N ∈ (1, 2)) ? sum(size.(k.ks_diag, 1)) : 1
size(k::CatKernel) = (size(k, 1), size(k, 1))

function cov(k::CatKernel, X::AV{<:AVM})
    Σ = BlockMatrix{Float64}(uninitialized_blocks, size.(X, 1), size.(X, 1))
    for q in eachindex(k.ks_diag)
        setblock!(Σ, Matrix(cov(k.ks_diag[q], X[q])), q, q)
        for p in 1:q-1
            setblock!(Σ, xcov(k.ks_off[p, q], X[p], X[q]), p, q)
        end
    end
    return LazyPDMat(SquareDiagonal(Σ))
end
cov(k::CatKernel, X::AVM) = cov(k, [X])
function xcov(k::CatKernel, X::AV{<:AVM}, X′::AV{<:AVM})
    Ω = BlockMatrix{Float64}(uninitialized_blocks, size.(X, 1), size.(X′, 1))
    for q in eachindex(k.ks_diag), p in eachindex(k.ks_diag)
        if p == q
            setblock!(Ω, xcov(k.ks_diag[p], X[p], X′[p]), p, p)
        elseif p < q
            setblock!(Ω, xcov(k.ks_off[p, q], X[p], X′[q]), p, q)
        else
            setblock!(Ω, xcov(k.ks_off[q, p], X[p], X′[q]), p, q)
        end
    end
    return Ω
end

xcov(k::Union{<:CatCrossKernel, <:CrossKernel}, X::AVM, X′::AVM) = xcov(k, [X], [X′])
xcov(k::Union{<:CatCrossKernel, <:CrossKernel}, X::AV{<:AVM}, X′::AVM) = xcov(k, X, [X′])
xcov(k::Union{<:CatCrossKernel, <:CrossKernel}, X::AVM, X′::AV{<:AVM}) = xcov(k, [X], X′)
