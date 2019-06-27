"""
    cross(fs::AbstractVector{<:GP})

Creates a multi-output GP from an `AbstractVector` of `GP`s.
"""
function cross(fs::AbstractVector{<:GP})
    consistency_checks(fs)
    return GP(first(fs).gpc, cross, fs)
end

function consistency_checks(fs)
    @assert length(fs) >= 1
    @assert all([f.gpc == first(fs).gpc for f in fs])
end
Zygote.@nograd consistency_checks

μ_p′(::typeof(cross), fs) = BlockMean(mean.(fs))

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
