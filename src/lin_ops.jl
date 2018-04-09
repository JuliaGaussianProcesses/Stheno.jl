import Base: +, *, |
export ←, |

const μ = mean_function
const k = kernel

"""
    (f_q::GP)(X::AbstractMatrix)

A GP evaluated at `X` is a multivariate finite-dimensional GP whose mean and covariance
are fully specified by `X` and the mean function and kernel of `f_q`.

    (f_q::GP{<:Finite})(X::AbstractMatrix)

A GP on a finite-dimensional index set (i.e. a multivariate Normal) indexed at locations `X`
is a new GP with finite-dimensional index set whose cardinality is at most that of `f_q`.

Note that nested indexing is currently not supported.
"""
function (f_q::GP)(X::AM)
    isfinite(f_q) && throw(error("Nested indexing not currently supported."))
    return GP(f_q, X)
end
μ_p′(f_q::GP, X::AM) = FiniteMean(μ(f_q), X)
k_p′(f_q::GP, X::AM) = FiniteKernel(k(f_q), X)
function k_p′p(f_p::GP, f_q::GP, X::AM)
    kqp = k(f_q, f_p)
    return isfinite(f_p) ? FiniteCrossKernel(kqp, X) : LhsFiniteCrossKernel(kqp, X)
end
function k_pp′(f_p::GP, f_q::GP, X::AM)
    kpq = kernel(f_p, f_q)
    return isfinite(f_p) ? FiniteCrossKernel(kpq, X) : RhsFiniteCrossKernel(kpq, X)
end
length(::GP, X::AM) = length(X)

"""
    |(g::GP, c::Observation...)

`|` is NOT bit-wise logical OR in this context, it is the conditioning operator. That is, it
returns the conditional (posterior) distribution over everything on the left given the
`Observation`(s) on the right.
"""
|(g::GP, c::Observation) = g | (c,)
function |(g::GP, c::Tuple{Vararg{Observation}})
    f, y = [getfield.(c, :f)...], vcat(getfield.(c, :y)...)
    @show f, μ.(f)
    μf, kff = CatMean(μ.(f)), CatKernel(k.(f), k.(f, permutedims(f)))
    return GP(|, g, f, CondCache(kff, μf, y))
end
μ_p′(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache) =
    ConditionalMean(cache, μ(g), CatCrossKernel(kernel.(f, Ref(g))))
k_p′(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache) =
    ConditionalKernel(cache, CatCrossKernel(kernel.(f, Ref(g))), kernel(g))
function k_p′p(g::GP, ::typeof(|), h::GP, f::Vector{<:GP}, cache::CondCache)
    kfg, kfh = CatCrossKernel(kernel.(f, Ref(g))), CatCrossKernel(kernel.(f, Ref(h)))
    return ConditionalCrossKernel(cache, kfg, kfh, kernel(g, h))
end
k_pp′(h::GP, ::typeof(|), g::GP, f::Vector{<:GP}, c::CondCache) = k_p′p(g, |, h, f, c)
length(::typeof(|), f::GP, ::GP, f̂::Vector) = length(f)
