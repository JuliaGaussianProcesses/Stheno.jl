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
    return mean_vector(g, x) + cov(g, fx.f, x, fx.x) * α
end

function cov((_, g, C_ff, _, fx, _)::cond_data, x::AV)
    return cov(g, x) - Xt_invA_X(C_ff, cov(fx.f, g, fx.x, x))
end
function cov_diag((_, g, C, _, fx, _)::cond_data, x::AV)
    return cov_diag(g, x) - diag_Xt_invA_X(C, cov(fx.f, g, fx.x, x))
end

function cov((_, g, C_ff, _, fx, _)::cond_data, x::AV, x′::AV)
    X, Y = cov(fx.f, g, fx.x, x), cov(fx.f, g, fx.x, x′)
    return cov(g, x, x′) - Xt_invA_Y(X, C_ff, Y)
end
function cov_diag((_, g, C_ff, _, fx, _)::cond_data, x::AV, x′::AV)
    X, Y = cov(fx.f, g, fx.x, x), cov(fx.f, g, fx.x, x′)
    return cov_diag(g, x, x′) - diag_Xt_invA_Y(X, C_ff, Y)
end

function cov((_, g, C, _, fx, _)::cond_data, f′::AbstractGP, x::AV, x′::AV)
    X, Y = cov(fx.f, g, fx.x, x), cov(fx.f, f′, fx.x, x′)
    return cov(g, f′, x, x′) - Xt_invA_Y(X, C, Y)
end

function cov(f::AbstractGP, (_, g, C, _, fx, _)::cond_data, x::AV, x′::AV)
    X, Y = cov(fx.f, f, fx.x, x), cov(fx.f, g, fx.x, x′)
    return cov(f, g, x, x′) - Xt_invA_Y(X, C, Y)
end

function cov_diag((_, g, C, _, fx, _)::cond_data, f′::AbstractGP, x::AV, x′::AV)
    X, Y = cov(fx.f, g, fx.x, x), cov(fx.f, f′, fx.x, x′)
    return cov_diag(g, f′, x, x′) - diag_Xt_invA_Y(X, C, Y)
end

function cov_diag(f::AbstractGP, (_, g, C, _, fx, _)::cond_data, x::AV, x′::AV)
    X, Y = cov(fx.f, f, fx.x, x), cov(fx.f, g, fx.x, x′)
    return cov_diag(f, g, x, x′) - diag_Xt_invA_Y(X, C, Y)
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
|(g::AbstractGP, c::Tuple{Vararg{Observation}}) = g | merge(c)

|(g::Tuple{Vararg{AbstractGP}}, c::Tuple{Vararg{Observation}}) = map(g_ -> g_ | c, g)

|(g::Tuple{Vararg{AbstractGP}}, c::Observation) = g | (c,)
