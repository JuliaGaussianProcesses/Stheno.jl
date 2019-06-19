###################### Implementation of AbstractGaussianProcess interface #################

"""
    BlockGP{Tfs<:AbstractVector{<:AbstractGP}} <: AbstractGP

A collection of `AbstractGaussianProcess`es. Akin to the usual conception of a
Multi-Output Gaussian process.
"""
struct BlockGP{Tfs<:AV{<:AbstractGP}} <: AbstractGP
    fs::Tfs
end
BlockGP(fs::Tuple{<:AbstractGP}) = BlockGP([fs...])
BlockGP(fs::AbstractGP...) = BlockGP(fs)

mean(f::BlockGP) = BlockMean(mean.(f.fs))
kernel(f::BlockGP) = BlockKernel(_kernels(f.fs, permutedims(f.fs)))
kernel(f1::BlockGP, f2::BlockGP) = BlockCrossKernel(_kernels(f1.fs, permutedims(f2.fs)))
kernel(f1::GP, f2::BlockGP) = BlockCrossKernel(kernel.(Ref(f1), permutedims(f2.fs)))
kernel(f1::BlockGP, f2::GP) = BlockCrossKernel(kernel.(f1.fs, Ref(f2)))

deconstruct(f::BlockGP) = (f.fs...,)

# Return a gpc object from one of the children.
getproperty(f::BlockGP, d::Symbol) = d === :gpc ? f.fs[1].gpc : getfield(f, d)

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
