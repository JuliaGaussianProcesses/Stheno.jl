import Base: +, *, |
export ←, |

const μ = mean
const k = kernel

"""
    (f_q::GP)(X::AbstractMatrix)

A GP evaluated at `X` is a multivariate finite-dimensional GP whose mean and covariance
are fully specified by `X` and the mean function and kernel of `f_q`.
"""
(f_q::GP)(X::AVM) = GP(f_q, X)
μ_p′(f_q::GP, X::AVM) = FiniteMean(μ(f_q), X)
k_p′(f_q::GP, X::AVM) = FiniteKernel(k(f_q), X)
k_p′p(f_p::GP, f_q::GP, X::AVM) = LhsFiniteCrossKernel(k(f_q, f_p), X)
k_pp′(f_p::GP, f_q::GP, X′::AVM) = RhsFiniteCrossKernel(k(f_p, f_q), X′)
length(::GP, X::AVM) = length(X)

"""
    |(g::GP, c::Observation...)

`|` is NOT bit-wise logical OR in this context, it is the conditioning operator. That is, it
returns the conditional (posterior) distribution over everything on the left given the
`Observation`(s) on the right.
"""
|(g::GP, c::Observation) = ((g,) | (c,))[1]
|(g::GP, c::Tuple{Vararg{Observation}}) = ((g,) | c)[1]
|(g::Tuple{Vararg{<:GP}}, c::Observation) = g | (c,)
function |(g::Tuple{Vararg{<:GP}}, c::Tuple{Vararg{Observation}})
    f, y = [getfield.(c, :f)...], BlockVector([getfield.(c, :y)...])
    @show f
    f_qs, Xs = [f_.f for f_ in f], [f_.args[1] for f_ in f]
    @show f_qs
    μf, kff = CatMean(μ.(f_qs)), CatKernel(k.(f_qs), k.(f_qs, permutedims(f_qs)))
    cache = CondCache(kff, μf, Xs, y)
    return map(g_->GP(|, g_, f_qs, cache), g)
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
