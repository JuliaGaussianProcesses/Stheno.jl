import Base: |, merge
export ←, |

"""
    Observation

Represents fixing a paricular (finite) GP to have a particular (vector) value.
"""
struct Observation{Tf<:FiniteGP, Ty<:AbstractVector}
    f::Tf
    y::Ty
end

const Obs = Observation
export Obs

←(f, y) = Observation(f, y)
get_f(c::Observation) = c.f
get_y(c::Observation) = c.y
function merge(c::Union{AbstractVector{<:Observation}, Tuple{Vararg{Observation}}})
    block_gp = BlockGP([map(c->get_f(c).f, c)...])
    block_x = BlockData([map(c->get_f(c).x, c)...])
    block_σ² = BlockVector([map(c->get_f(c).σ², c)...])
    block_y = BlockVector([map(get_y, c)...])
    return FiniteGP(block_gp, block_x, block_σ²)←block_y
end

"""
    |(g::AbstractGP, c::Observation)

Condition `g` on observation `c`.
"""
function |(g::GP, c::Observation)
    f, x, y, σ² = c.f.f, c.f.x, c.y, c.f.σ²
    return GP(|, g, f, CondCache(kernel(f), mean(f), x, y, σ²))
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
