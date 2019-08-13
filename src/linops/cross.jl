"""
    cross(fs::AbstractVector{<:GP})

Creates a multi-output GP from an `AbstractVector` of `GP`s.
"""
function cross(fs::AbstractVector{<:GP})
    consistency_checks(fs)
    return CompositeGP((cross, fs), first(fs).gpc)
end

function consistency_checks(fs)
    @assert length(fs) >= 1
    @assert all([f.gpc == first(fs).gpc for f in fs])
end
Zygote.@nograd consistency_checks


const cross_args{T<:AbsractVector{<:AbstractGP}} = Tuple{typeof(cross), T}

function mean_vector((_, fs)::cross_args, x::BlockData)
    blks = map((f, blk)->mean_vector(f, blk), fs, blocks(x))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
end

function cov((_, fs)::cross_args, x::BlockData)
    
end

function cov((_, fs)::cross_args, x::BlockData, x′::BlockData)

end

function cov_diag((_, fs)::cross_args, x::BlockData)

end

function xcov((_, fs)::cross_args, f′::AbstractGP, x::BlockData, x′::BlockData)

end
function xcov(f::AbstractGP, (_, fs)::cross_args, x::BlockData, x′::BlockData)

end

function xcov_diag((_, fs)::cross_args, f′::AbstractGP, x::BlockData)

end
function xcov_diag(f::AbstractGP, (_, fs)::cross_args, x::BlockData)

end


function ew(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    blks = map((k, b, b′)->ew(k, b, b′), diag(k.ks), blocks(x), blocks(x′))
    return Vector(_BlockArray(blks, _get_block_sizes(blks)...))
end
function pw(k::BlockCrossKernel, x::BlockData, x′::BlockData)
    return Matrix(_pw(k.ks, blocks(x), permutedims(blocks(x′))))
end

pw(k::BlockCrossKernel, x::BlockData, x′::AV) = pw(k, x, BlockData([x′]))
pw(k::BlockCrossKernel, x::AV, x′::BlockData) = pw(k, BlockData([x]), x′)


# μ_p′(::typeof(cross), fs) = BlockMean(mean.(fs))

k_p′(::typeof(cross), fs) = BlockKernel(_kernels(fs, permutedims(fs)))

function k_pp′(fp::GP{<:MeanFunction, <:BlockKernel}, ::typeof(cross), fs)
    return BlockCrossKernel(_kernels(last(fp.args), permutedims(fs)))
end

k_pp′(fp::GP, ::typeof(cross), fs) = BlockCrossKernel(_kernels(Ref(fp), permutedims(fs)))

function k_p′p(::typeof(cross), fs, fp::GP{<:MeanFunction, <:BlockKernel})
    return BlockCrossKernel(_kernels(fs, permutedims(last(fp.args))))
end

k_p′p(::typeof(cross), fs, fp::GP) = BlockCrossKernel(kernel.(fs, Ref(fp)))

# This is a helper function to ensure that Zygote-based AD can be employed.
_kernels(fs1, fs2) = kernel.(fs1, fs2)
@adjoint function _kernels(fs1, fs2)
    ys_and_backs = broadcast((f1, f2)->Zygote.forward(kernel, f1, f2), fs1, fs2)
    ys, backs = first.(ys_and_backs), last.(ys_and_backs)
    function back(Δ::Diagonal{T}) where {T}
        back_mat = Matrix{Union{T, Nothing}}(undef, size(Δ))
        fill!(back_mat, nothing)
        back_mat[diagind(Δ)] = diag(Δ)
        return back(back_mat)
    end
    function back(Δ)
        Δ_fs1_fs2 = broadcast((back, δ)->back(δ), backs, Δ)

        Δ_fs1, Δ_fs2 = first.(Δ_fs1_fs2), last.(Δ_fs1_fs2)
        δ_1 = Vector{Any}(undef, length(fs1))
        δ_2 = Vector{Any}(undef, length(fs2))

        for p in 1:length(δ_1)
            δ_1[p] = Δ_fs1[p, 1]
            for q in 2:length(δ_2)
                δ_1[p] = Zygote.accum(δ_1[p], Δ_fs1[p, q])
            end
        end

        for q in 1:length(δ_2)
            δ_2[q] = Δ_fs2[1, q]
            for p in 2:length(δ_1)
                δ_2[q] = Zygote.accum(δ_2[q], Δ_fs2[p, q])
            end
        end

        return δ_1, reshape(δ_2, 1, :)
    end
    return ys, back
end


#
# Helper for pw.
#

function _pw(ks, x_blks, x′_blks)
    blks = pw.(ks, x_blks, x′_blks)
    return _BlockArray(blks, _get_block_sizes(blks)...)
end
@adjoint function _pw(ks, x_blks, x′_blks)
    blk_backs = broadcast((k, x, x′)->Zygote.forward(pw, k, x, x′), ks, x_blks, x′_blks)
    blks, backs = first.(blk_backs), last.(blk_backs)
    Y = _BlockArray(blks, _get_block_sizes(blks)...)

    function back(Δ::BlockMatrix)

        Δ_k_x_x′ = broadcast((back, blk)->back(blk), backs, Δ.blocks)
        Δ_ks, Δ_x, Δ_x′ = first.(Δ_k_x_x′), getindex.(Δ_k_x_x′, 2), getindex.(Δ_k_x_x′, 3)

        # Reduce over appropriate dimensions manually because sum doesn't work... :S
        δ_x = Vector{Any}(undef, length(x_blks))
        δ_x′ = Vector{Any}(undef, length(x′_blks))

        for p in 1:length(x_blks)
            δ_x[p] = Δ_x[p, 1]
            for q in 2:length(x′_blks)
                δ_x[p] = Zygote.accum(δ_x[p], Δ_x[p, q])
            end
        end

        for q in 1:length(x′_blks)
            δ_x′[q] = Δ_x′[1, q]
            for p in 2:length(x_blks)
                δ_x′[q] = Zygote.accum(δ_x′[q], Δ_x′[p, q])
            end
        end

        return Δ_ks, δ_x, permutedims(δ_x′)
    end
    return Y, back
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
