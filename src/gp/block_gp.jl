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
        Δ_fs1 = reduce(Zygote.accum, first.(Δ_fs1_fs2); dims=2, init=nothing)
        Δ_fs2 = reduce(Zygote.accum, last.(Δ_fs1_fs2); dims=1, init=nothing)
        return Δ_fs1, Δ_fs2
    end
end
