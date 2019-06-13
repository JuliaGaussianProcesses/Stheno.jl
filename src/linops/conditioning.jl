import Base: |, merge
export ←, |

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value.
"""
struct Observation{Tf<:FiniteGP, Ty<:Vector}
    f::Tf
    y::Ty
end

const Obs = Observation
export Obs

←(f, y) = Observation(f, y)
get_f(c::Observation) = c.f
get_y(c::Observation) = c.y

function merge(fs::Tuple{Vararg{FiniteGP}})
    block_gp = BlockGP([map(f->f.f, fs)...])
    block_x = BlockData([map(f->f.x, fs)...])
    block_Σy = block_diagonal([map(f->f.Σy, fs)...])
    return FiniteGP(block_gp, block_x, block_Σy)
end
function merge(c::Tuple{Vararg{Observation}})
    block_y = Vector(BlockVector([map(get_y, c)...]))
    return merge(map(get_f, c))←block_y
end

"""
    |(g::AbstractGP, c::Observation)

Condition `g` on observation `c`.
"""
function |(g::GP, c::Observation)
    f, x, y, Σy = c.f.f, c.f.x, c.y, c.f.Σy
    return GP(|, g, f, CondCache(kernel(f), mean(f), x, y, Σy))
end
|(g::BlockGP, c::Observation) = BlockGP(g.fs .| Ref(c))

function μ_p′(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return iszero(kernel(f, g)) ? mean(g) : CondMean(cache, mean(g), kernel(f, g))
end
function k_p′(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return iszero(kernel(f, g)) ? kernel(g) : CondKernel(cache, kernel(f, g), kernel(g))
end
function k_p′p(::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache, h::AbstractGP)
    return iszero(kernel(f, g)) || iszero(kernel(f, h)) ?
        kernel(g, h) : CondCrossKernel(cache, kernel(f, g), kernel(f, h), kernel(g, h))
end
function k_pp′(h::AbstractGP, ::typeof(|), g::AbstractGP, f::AbstractGP, cache::CondCache)
    return iszero(kernel(f, g)) || iszero(kernel(f, h)) ?
        kernel(h, g) : CondCrossKernel(cache, kernel(f, h), kernel(f, g), kernel(h, g))
end

# Sugar
|(g::AbstractGP, c::Tuple{Vararg{Observation}}) = g | merge(c)
|(g::Tuple{Vararg{AbstractGP}}, c::Observation) = deconstruct(BlockGP([g...]) | c)
function |(g::Tuple{Vararg{AbstractGP}}, c::Tuple{Vararg{Observation}})
    return deconstruct(BlockGP([g...]) | merge(c))
end
