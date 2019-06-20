"""
    cross(fs::AbstractVector{<:GP})

Make something a bit like a multi-output GP.
"""
cross(fs::AbstractVector{<:GP}) = GP(cross, fs)

μ_p′(::typeof(cross), fs) = BlockMean(mean.(fs))

k_p′(::typeof(cross), fs) = BlockKernel(_kernels(f.fs, permutedims(f.fs)))

function k_pp′(fp::GP{<:Any, <:BlockKernel}, ::typeof(cross), fs)
    return BlockCrossKernel(_kernels(fp.args[2:end], permutedims(fs)))
end

k_pp′(fp::GP, ::typeof(cross), fs) = BlockCrossKernel(kernel.(Ref(f1), permutedims(f2.fs)))

function k_p′p(::typeof(cross), fs, fp::GP{<:Any, <:BlockKernel})
    return BlockCrossKernel(_kernels(fs, permutedims(fp.args[2:end])))
end

k_p′p(::typeof(cross), fs, fp::GP) = BlockCrossKernel(kernel.(f1.fs, Ref(f2)))

_kernels(fs1, fs2) = kernel.(fs1, fs2)
@adjoint function _kernels(fs1, fs2)
    ys_and_backs = broadcast((f1, f2)->Zygote.forward(kernel, f1, f2), fs1, fs2)
    ys, backs = first.(ys_and_backs), last.(ys_and_backs)
    return ys, function(Δ)
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
end
