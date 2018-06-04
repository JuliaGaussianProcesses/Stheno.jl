import Base: |
export ←, |

# """
#     |(g::Union{GP, Tuple{Vararg{GP}}}, c::Union{Observation, Tuple{Vararg{Observation}}})

# `|` is NOT bit-wise logical OR in this context, it is the conditioning operator. That is, it
# returns the conditional (posterior) distribution over everything on the left given the
# `Observation`(s) on the right.
# """
# |(g::GP, c::Observation) = ((g,) | (c,))[1]
# function |(g::GP, c::Observation)
#     f, y = c.f, c.y
#     f_q, X = f.args[1], f.args[2]
#     μf, kff = mean(f_q), kernel(f_q)
#     cache = CondCache(kff, μf, X, y)
#     return GP(|, g, f_q, cache)
# end
# μ_p′(::typeof(|), g::GP, f::GP, cache::CondCache) =
#     ConditionalMean(cache, mean(g), kernel(f, g))
# k_p′(::typeof(|), g::GP, f::GP, cache::CondCache) =
#     ConditionalKernel(cache, kernel(f, g), kernel(g))
# k_p′p((_, g, f, cache)::Tuple{typeof(|), GP, GP, CondCache}, h::GP) =
#     ConditionalCrossKernel(cache, kernel(f, g), kernel(f, h), kernel(g, h))
# k_pp′(h::GP, (_, g, f, cache)::Tuple{typeof(|), GP, GP, CondCache}) =
#     ConditionalCrossKernel(cache, kernel(f, h), kernel(f, g), kernel(h, g))

# All of the code below is a stop-gap while I'm figuring out what to do about the
# concatenation of GPs.
|(g::GP, c::Observation) = ((g,) | (c,))[1]
|(g::GP, c::Tuple{Vararg{Observation}}) = ((g,) | c)[1]
|(g::Tuple{Vararg{GP}}, c::Observation) = g | (c,)
function |(g::Tuple{Vararg{GP}}, c::Tuple{Vararg{Observation}})
    f, y = [getfield.(c, :f)...], BlockVector([getfield.(c, :y)...])
    f_qs, Xs = [f_.args[1] for f_ in f], [f_.args[2] for f_ in f]
    μf = CatMean(mean.(f_qs))
    kff = CatKernel(kernel.(f_qs), kernel.(f_qs, permutedims(f_qs)))
    return map(g_->GP(|, g_, f_qs, CondCache(kff, μf, Xs, y)), g)
end
function μ_p′(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache)
    return ConditionalMean(cache, mean(g), CatCrossKernel(kernel.(f, Ref(g))))
end
function k_p′(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache)
    return ConditionalKernel(cache, CatCrossKernel(kernel.(f, Ref(g))), kernel(g))
end
function k_p′p(::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache, h::GP)
    kfg, kfh = CatCrossKernel(kernel.(f, Ref(g))), CatCrossKernel(kernel.(f, Ref(h)))
    return ConditionalCrossKernel(cache, kfg, kfh, kernel(g, h))
end
function k_pp′(h::GP, ::typeof(|), g::GP, f::Vector{<:GP}, cache::CondCache)
    kfh, kfg = CatCrossKernel(kernel.(f, Ref(h))), CatCrossKernel(kernel.(f, Ref(g)))
    return ConditionalCrossKernel(cache, kfh, kfg, kernel(h, g))
end
length(::typeof(|), f::GP, ::GP, f̂::Vector) = length(f)
