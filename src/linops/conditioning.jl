function merge(fs::Tuple{Vararg{FiniteGP}})
    block_gp = cross([map(f->f.f, fs)...])
    block_x = BlockData([map(f->f.x, fs)...])
    block_Σy = block_diagonal([map(f->f.Σy, fs)...])
    return FiniteGP(block_gp, block_x, block_Σy)
end
function merge(c::Tuple{Vararg{Observation}})
    block_y = Vector(BlockVector([map(get_y, c)...]))
    return merge(map(get_f, c))←block_y
end

"""
    |(g::GP, c::Observation)

Condition `g` on observation `c`.
"""
function |(g::GP, c::Observation)
    f, x, y, Σy = c.f.f, c.f.x, c.y, c.f.Σy
    return GP(g.gpc, |, g, f, CondCache(kernel(f), mean(f), x, y, Σy))
end

function μ_p′(::typeof(|), g::GP, f::GP, cache::CondCache)
    return iszero(kernel(f, g)) ? mean(g) : CondMean(cache, mean(g), kernel(f, g))
end
function k_p′(::typeof(|), g::GP, f::GP, cache::CondCache)
    return iszero(kernel(f, g)) ? kernel(g) : CondKernel(cache, kernel(f, g), kernel(g))
end
function k_p′p(::typeof(|), g::GP, f::GP, cache::CondCache, h::GP)
    return iszero(kernel(f, g)) || iszero(kernel(f, h)) ?
        kernel(g, h) : CondCrossKernel(cache, kernel(f, g), kernel(f, h), kernel(g, h))
end
function k_pp′(h::GP, ::typeof(|), g::GP, f::GP, cache::CondCache)
    return iszero(kernel(f, g)) || iszero(kernel(f, h)) ?
        kernel(h, g) : CondCrossKernel(cache, kernel(f, h), kernel(f, g), kernel(h, g))
end

# Multi-arg stuff
|(g::GP, c::Tuple{Vararg{Observation}}) = g | merge(c)

|(g::Tuple{Vararg{GP}}, c::Tuple{Vararg{Observation}}) = map(g_ -> g_ | c, g)

|(g::Tuple{Vararg{GP}}, c::Observation) = g | (c,)

