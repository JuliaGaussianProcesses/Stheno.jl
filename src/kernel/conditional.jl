import Base: size, show
export Conditional

# Internal data structure used to cache various quantities to prevent recomputation.
struct ConditionalData
    U::UpperTriangular
    idx::Vector{Int}
    tmp::Vector{Float64}
    tmp′::Vector{Float64}
end
ConditionalData(U::UpperTriangular) =
    ConditionalData(U, collect(1:size(U, 1)), Vector{Float64}(size(U, 1)), Vector{Float64}(size(U, 1)))
==(a::ConditionalData, b::ConditionalData) = a.U == b.U && a.idx == b.idx

"""
    Conditional{Tk} <: Kernel{NonStationary}

A kernel for use in conditional distributions.
"""
struct Conditional{Tk} <: Kernel{NonStationary}
    k_ff′::Tk
    k_f̂f::Any
    k_f̂f′::Any
    data::ConditionalData
end
function (k::Conditional)(x, x′)
    a = At_ldiv_B!(k.data.U, broadcast!(k.k_f̂f, k.data.tmp, k.data.idx, x))
    b = At_ldiv_B!(k.data.U, broadcast!(k.k_f̂f′, k.data.tmp′, k.data.idx, x′))
    return k.k_ff′(x, x′) - dot(a, b)
end
==(a::Conditional{Tk}, b::Conditional{Tk}) where Tk =
    a.k_ff′ == b.k_ff′ && a.k_ff̂ == b.k_ff̂ && a.k_f̂f′ == b.k_f̂f′ && a.data == b.data
dims(k::Conditional) = dims(k.k_ff′)
size(k::Conditional) = size(k.k_ff′)
size(k::Conditional, n::Int) = size(k.k_ff′, n)
show(io::IO, k::Conditional) =
    print(io, "Conditional with prior kernel $(k.k_ff′)")
isfinite(k::Conditional) = isfinite(k.k_ff′)
