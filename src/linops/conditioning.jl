import Base: |
export ←, |

"""
    |(g::Union{GP, Tuple{Vararg{GP}}}, c::Union{Observation, Tuple{Vararg{Observation}}})

`|` is NOT bit-wise logical OR in this context, it is the conditioning operator. That is, it
returns the conditional (posterior) distribution over everything on the left given the
`Observation`(s) on the right.
"""
|(g::GP, c::Observation) = ((g,) | (c,))[1]
|(g::GP, c::Tuple{Vararg{Observation}}) = ((g,) | c)[1]
|(g::Tuple{Vararg{GP}}, c::Observation) = g | (c,)
function |(g::Tuple{Vararg{GP}}, c::Tuple{Vararg{Observation}})
    f, y = [getfield.(c, :f)...], BlockVector([getfield.(c, :y)...])
    f_qs, Xs = [f_.f for f_ in f], [f_.args[1] for f_ in f]
    μf = CatMean(mean.(f_qs))
    kff = CatKernel(kernel.(f_qs), kernel.(f_qs, permutedims(f_qs)))
    cache = CondCache(kff, μf, Xs, y)
    return map(g_->GP(|, g_, f_qs, cache), g)
end
μ_p′(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache) =
    ConditionalMean(cache, mean(g), CatCrossKernel(kernel.(f, Ref(g))))
k_p′(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache) =
    ConditionalKernel(cache, CatCrossKernel(kernel.(f, Ref(g))), kernel(g))
function k_p′p(g::GP, ::typeof(|), h::GP, f::Vector{<:GP}, cache::CondCache)
    kfg, kfh = CatCrossKernel(kernel.(f, Ref(g))), CatCrossKernel(kernel.(f, Ref(h)))
    return ConditionalCrossKernel(cache, kfg, kfh, kernel(g, h))
end
k_pp′(h::GP, ::typeof(|), g::GP, f::Vector{<:GP}, c::CondCache) = k_p′p(g, |, h, f, c)
length(::typeof(|), f::GP, ::GP, f̂::Vector) = length(f)
