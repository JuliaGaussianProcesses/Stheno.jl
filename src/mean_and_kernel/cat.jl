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
# mean(μ::CatMean, X::Vector{<:AbstractMatrix}) = 
mean(μ::CatMean) = vcat(mean.(μ.μ)...)
isfinite(μ::CatMean) = all(isfinite.(μ.μ))
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
    function CatCrossKernel(ks::Matrix)
        @assert all(isfinite.(ks))
        if size(ks, 2) > 1
            for p in 1:size(ks, 1)
                @assert reduce((k, k′)->size(k, 1) == size(k′, 1), view(ks, p, :))
            end
        end
        if size(ks, 1) > 1
            for q in 1:size(ks, 2)
                @assert reduce((k, k′)->size(k, 2) == size(k′, 2), view(ks, :, q))
            end
        end
        return new(ks)
    end
end
CatCrossKernel(ks::Vector) = CatCrossKernel(reshape(ks, length(ks), 1))
CatCrossKernel(ks::RowVector) = CatCrossKernel(reshape(ks, 1, length(ks)))
size(k::CatCrossKernel) = (size(k, 1), size(k, 2))
size(k::CatCrossKernel, N::Int) = N == 1 ?
    sum(size.(k.ks[:, 1], Ref(1))) :
    N == 2 ? sum(size.(k.ks[1, :], Ref(2))) : 1

"""
    xcov(k::CatCrossKernel)

Get the xcov matrix of a (finite) CatCrossKernel. Currently uses a dense representation,
which is problematic from the perspective of retaining structure. Will need change over to
use `BlockArray`s for efficiency.
"""
function xcov(k::CatCrossKernel)
    Ω = Matrix{Float64}(undef, size(k))
    rs = vcat(0, cumsum(size.(k.ks[:, 1], Ref(1))))
    cs = vcat(0, cumsum(size.(k.ks[1, :], Ref(2))))
    for I in CartesianIndices(k.ks)
        Ω[rs[I[1]]+1:rs[I[1]+1], cs[I[2]]+1:cs[I[2]+1]] = xcov(k.ks[I[1], I[2]])
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

"""
    cov(k::CatKernel)

Get the covariance matrix of a (finite) CatKernel. Currently uses a dense representation,
which is problematic from the perspective of retaining structure. Will need change over to
use `BlockArray`s for efficiency.
"""
function cov(k::CatKernel)
    Σ, rs = Matrix{Float64}(undef, size(k)), vcat(0, cumsum(size.(k.ks_diag, Ref(1))))
    for c in eachindex(k.ks_diag)
        Σ[rs[c]+1:rs[c+1], rs[c]+1:rs[c+1]] = Matrix(cov(k.ks_diag[c]))
        for r in 1:c-1
            Σ[rs[r]+1:rs[r+1], rs[c]+1:rs[c+1]] = xcov(k.ks_off[r, c])
        end
    end
    return LazyPDMat(Symmetric(Σ))
end
