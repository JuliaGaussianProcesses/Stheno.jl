export PosteriorKernel

# Internal data structure used to cache various quantities to prevent recomputation.
struct PosteriorData
    U::UpperTriangular
    idx::Vector{Int}
    tmp::Vector{Float64}
    tmp′::Vector{Float64}
end
PosteriorData(U::UpperTriangular) =
    PosteriorData(U, collect(1:size(U, 1)), Vector{Float64}(size(U, 1)), Vector{Float64}(size(U, 1)))
==(a::PosteriorData, b::PosteriorData) = a.U == b.U && a.idx == b.idx

"""
    PosteriorKernel{Tk} <: Kernel{NonStationary}

A kernel for use in posterior distributions. Currently doesn't provide a particularly
efficient implementation, but this will change in the future.
"""
struct PosteriorKernel{Tk} <: Kernel{NonStationary}
    k_ff′::Tk
    k_f̂f::Any
    k_f̂f′::Any
    data::PosteriorData
end
function (k::PosteriorKernel)(x, x′)
    a = At_ldiv_B!(k.data.U, broadcast!(k.k_f̂f, k.data.tmp, k.data.idx, x))
    b = At_ldiv_B!(k.data.U, broadcast!(k.k_f̂f′, k.data.tmp′, k.data.idx, x′))
    return k.k_ff′(x, x′) - dot(a, b)
end
==(a::PosteriorKernel{Tk}, b::PosteriorKernel{Tk}) where Tk =
    a.k_ff′ == b.k_ff′ && a.k_ff̂ == b.k_ff̂ && a.k_f̂f′ == b.k_f̂f′ && a.data == b.data
dims(k::PosteriorKernel) = dims(k.k_ff′)
