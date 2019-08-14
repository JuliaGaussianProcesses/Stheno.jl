using LinearAlgebra: Cholesky

"""
    |(g::GP, c::Observation)

Condition `g` on observation `c`.
"""
function |(g::AbstractGP, c::Observation)
    C = cholesky(Symmetric(cov(c.f)))
    return CompositeGP((|, g, C, C \ (c.y - mean(c.f)), c.f, c.y), g.gpc)
end

const cond_data = Tuple{typeof(|), AbstractGP, Cholesky, AV{<:Real}, FiniteGP, AV{<:Real}}

function mean_vector((_, g, _, α, fx, _)::cond_data, x::AV)
    return mean(g(x)) + cov(g(x), fx) * α
end

function cov((_, g, C_ff, _, fx, _)::cond_data, x::AV)
    return cov(g(x)) - Xt_invA_X(C_ff, cov(fx, g(x)))
end

function cov((_, g, C_ff, _, fx, _)::cond_data, x::AV, x′::AV)
    return cov(g(x), g(x′)) - Xt_invA_Y(cov(fx, g(x)), C_ff, cov(fx, g(x′)))
end

function cov_diag((_, g, C, _, fx, _)::cond_data, x::AV)
    return cov_diag(g, x) - diag_Xt_invA_X(C, cov(fx, g(x)))
end

function xcov((_, g, C, _, fx, _)::cond_data, f′::AbstractGP, x::AV, x′::AV)
    return cov(g(x), f′(x)) - Xt_invA_Y(cov(fx, g(x)), C, cov(fx, f′(x′)))
end

function xcov(f::AbstractGP, (_, g, C, _, fx, _)::cond_data, x::AV, x′::AV)
    return cov(f(x), g(x′)) - Xt_invA_Y(cov(fx, f(x), C, cov(fx, g(x′))))
end

function xcov_diag((_, g, C, _, fx, _)::cond_data, f′::AbstractGP, x::AV)
    return xcov_diag(g, f′, x) - diag_Xt_invA_Y(cov(fx, g(x), C, cov(fx, f′(x′))))
end

function xcov_diag(f::AbstractGP, (_, g, C, _, fx, _)::cond_data, x::AV)
    return xcov_diag(f, g, x) - diag_Xt_invA_Y(cov(fx, f(x)), C, cov(fx, g(x)))
end

function sample(rng::AbstractRNG, (_, g, C, _, fx, y)::cond_data, x::AV, S::Int)
    X, Y = rand(rng, [g(x), fx], S)
    return X .+ cov(fx, g(x))' * (C \ (y .- Y))
end

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

# Multi-arg stuff
|(g::GP, c::Tuple{Vararg{Observation}}) = g | merge(c)

|(g::Tuple{Vararg{GP}}, c::Tuple{Vararg{Observation}}) = map(g_ -> g_ | c, g)

|(g::Tuple{Vararg{GP}}, c::Observation) = g | (c,)

