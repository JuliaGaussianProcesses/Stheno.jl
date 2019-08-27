"""
    cross(fs::AbstractVector{<:GP})

Creates a multi-output GP from an `AbstractVector` of `GP`s.
"""
function cross(fs::AbstractVector{<:AbstractGP})
    consistency_checks(fs)
    return CompositeGP((cross, fs), first(fs).gpc)
end

function consistency_checks(fs)
    @assert length(fs) >= 1
    @assert all([f.gpc == first(fs).gpc for f in fs])
end
Zygote.@nograd consistency_checks


const cross_args{T<:AbstractVector{<:AbstractGP}} = Tuple{typeof(cross), T}

function mean_vector((_, fs)::cross_args, x::BlockData)
    blks = map((f, blk)->mean_vector(f, blk), fs, blocks(x))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
end

function cov((_, fs)::cross_args, x::BlockData)
    Cs = reshape(map((f, blk)->cov(f, (cross, fs), blk, x), fs, blocks(x)), :, 1)
    return Matrix(_BlockArray(reshape(Cs, :, 1), _get_block_sizes(Cs)...))
end

function cov_diag((_, fs)::cross_args, x::BlockData)
    cs = map(cov_diag, fs, blocks(x))
    return Vector(_BlockArray(cs, _get_block_sizes(cs)...))
end

function cov((_, fs)::cross_args, x::BlockData, x′::BlockData)
    Cs = reshape(map((f, blk)->cov(f, (cross, fs), blk, x′), fs, blocks(x)), :, 1)
    return Matrix(_BlockArray(reshape(Cs, :, 1), _get_block_sizes(Cs)...))
end

function cov_diag((_, fs)::cross_args, x::BlockData, x′::BlockData)
    cs = map(cov_diag, fs, blocks(x), blocks(x′))
    return Vector(_BlockArray(cs, _get_block_sizes(cs)...))
end

function cov((_, fs)::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    Cs = reshape(map((f, x)->cov(f, f′, x, x′), fs, blocks(x)), :, 1)
    return Matrix(_BlockArray(Cs, _get_block_sizes(Cs)...))
end
function cov(f::AbstractGP, (_, fs)::cross_args, x::AV, x′::BlockData)
    Cs = reshape(map((f′, x′)->cov(f, f′, x, x′), fs, blocks(x′)), 1, :)
    return Matrix(_BlockArray(Cs, _get_block_sizes(Cs)...))
end

function cov_diag(args::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    return diag(cov(args, f′, x, x′))
end
function cov_diag(f::AbstractGP, args::cross_args, x::AV, x′::BlockData)
    return diag(cov(f, args, x, x′))
end



#
# Util for multi-process versions of `rand`, `logpdf`, and `elbo`.
#

function finites_to_block(fs::AV{<:FiniteGP})
    return FiniteGP(
        cross(map(f->f.f, fs)),
        BlockData(map(f->f.x, fs)),
        make_block_noise(map(f->f.Σy, fs)),
    )
end

make_block_noise(Σys::Vector{<:Diagonal}) = Diagonal(Vector(BlockVector(diag.(Σys))))
make_block_noise(Σys::Vector) = block_diagonal(Σys)

function _get_indices(fs)
    sz = cumsum(map(length, fs))
    return [sz[n] - length(fs[n]) + 1:sz[n] for n in eachindex(fs)]
end
Zygote.@nograd _get_indices


#
# multi-process `rand`
#

function rand(rng::AbstractRNG, fs::AV{<:FiniteGP}, N::Int)
    Y = rand(rng, finites_to_block(fs), N)
    idx = _get_indices(fs)
    return map(n->Y[idx[n], :], eachindex(fs))
end
rand(rng::AbstractRNG, fs::AV{<:FiniteGP}) = vec.(rand(rng, fs, 1))
rand(fs::AV{<:FiniteGP}, N::Int) = rand(Random.GLOBAL_RNG, fs, N)
rand(fs::AV{<:FiniteGP}) = vec.(rand(Random.GLOBAL_RNG, fs))


#
# multi-process `logpdf`
#

function logpdf(fs::AV{<:FiniteGP}, ys::Vector{<:AV{<:Real}})
    return logpdf(finites_to_block(fs), vcat(ys...))
end
logpdf(fs::Vector{<:Observation}) = logpdf(get_f.(fs), get_y.(fs))


#
# multi-process `elbo`
#

function elbo(fs::Vector{<:FiniteGP}, ys::Vector{<:AV{<:Real}}, us::Vector{<:FiniteGP})
    return elbo(finites_to_block(fs), Vector(BlockVector(ys)), finites_to_block(us))
end

function elbo(fs::Vector{<:FiniteGP}, ys::Vector{<:AV{<:Real}}, u::FiniteGP)
    return elbo(finites_to_block(fs), Vector(BlockVector(ys)), u)
end

elbo(f::FiniteGP, y::AV{<:Real}, us::Vector{<:FiniteGP}) = elbo(f, y, finites_to_block(us))
