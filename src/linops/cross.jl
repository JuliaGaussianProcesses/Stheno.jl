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
    Cs = xcov.(fs, reshape(fs, 1, :), blocks(x), reshape(blocks(x), 1, :))
    return Matrix(_BlockArray(Cs, _get_block_sizes(Cs)...))
end

function cov((_, fs)::cross_args, x::BlockData, x′::BlockData)
    Cs = xcov.(fs, reshape(fs, 1, :), blocks(x), reshape(blocks(x′), 1, :))
    return Matrix(_BlockArray(Cs, _get_block_sizes(Cs)...))
end

function cov_diag((_, fs)::cross_args, x::BlockData)
    cs = map(cov_diag, fs, blocks(x))
    return Vector(_BlockArray(cs, _get_block_sizes(cs)...))
end

function xcov((_, fs)::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    Cs = reshape(map((f, x)->xcov(f, f′, x, x′), fs, blocks(x)), :, 1)
    return Matrix(_BlockArray(Cs, _get_block_sizes(Cs)...))
end
function xcov(f::AbstractGP, (_, fs)::cross_args, x::AV, x′::BlockData)
    Cs = reshape(map((f′, x′)->xcov(f, f′, x, x′), fs, blocks(x′)), 1, :)
    return Matrix(_BlockArray(Cs, _get_block_sizes(Cs)...))
end

function xcov_diag(args::cross_args, f′::AbstractGP, x::BlockData)
    return diag(xcov(args, f′, x))
end
function xcov_diag(f::AbstractGP, args::cross_args, x::BlockData)
    return diag(xcov(f, args, x))
end

function sample(rng::AbstractRNG, (_, fs)::cross_args, x::BlockData, S::Int)
    blk = map((f, blk)->sample(rng, f, x, S), fs, blocks(x))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
end


#
# Helper for pw.
#

# function _xcov(fs, f′s, x_blks, x′_blks)
#     blks = pw.(ks, x_blks, x′_blks)
#     return _BlockArray(blks, _get_block_sizes(blks)...)
# end
# @adjoint function _pw(ks, x_blks, x′_blks)
#     blk_backs = broadcast((k, x, x′)->Zygote.forward(pw, k, x, x′), ks, x_blks, x′_blks)
#     blks, backs = first.(blk_backs), last.(blk_backs)
#     Y = _BlockArray(blks, _get_block_sizes(blks)...)

#     function back(Δ::BlockMatrix)

#         Δ_k_x_x′ = broadcast((back, blk)->back(blk), backs, Δ.blocks)
#         Δ_ks, Δ_x, Δ_x′ = first.(Δ_k_x_x′), getindex.(Δ_k_x_x′, 2), getindex.(Δ_k_x_x′, 3)

#         # Reduce over appropriate dimensions manually because sum doesn't work... :S
#         δ_x = Vector{Any}(undef, length(x_blks))
#         δ_x′ = Vector{Any}(undef, length(x′_blks))

#         for p in 1:length(x_blks)
#             δ_x[p] = Δ_x[p, 1]
#             for q in 2:length(x′_blks)
#                 δ_x[p] = Zygote.accum(δ_x[p], Δ_x[p, q])
#             end
#         end

#         for q in 1:length(x′_blks)
#             δ_x′[q] = Δ_x′[1, q]
#             for p in 2:length(x_blks)
#                 δ_x′[q] = Zygote.accum(δ_x′[q], Δ_x′[p, q])
#             end
#         end

#         return Δ_ks, δ_x, permutedims(δ_x′)
#     end
#     return Y, back
# end




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
