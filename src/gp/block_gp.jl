###################### Implementation of AbstractGaussianProcess interface #################

"""
    BlockGP{Tfs<:AbstractVector{<:AbstractGP}} <: AbstractGP

A collection of `AbstractGaussianProcess`es. Akin to the usual conception of a
Multi-Output Gaussian process.
"""
struct BlockGP{Tfs<:AV{<:AbstractGP}} <: AbstractGP
    fs::Tfs
end
BlockGP(fs::Tuple) = BlockGP([fs...])
BlockGP(fs...) = BlockGP(fs)

mean(f::BlockGP) = BlockMean(mean.(f.fs))
kernel(f::BlockGP) = BlockKernel(kernel.(f.fs, permutedims(f.fs)))
kernel(f1::BlockGP, f2::BlockGP) = BlockCrossKernel(kernel.(f1.fs, permutedims(f2.fs)))
kernel(f1::GP, f2::BlockGP) = BlockCrossKernel(kernel.(Ref(f1), permutedims(f2.fs)))
kernel(f1::BlockGP, f2::GP) = BlockCrossKernel(kernel.(f1.fs, Ref(f2)))

deconstruct(f::BlockGP) = (f.fs...,)

# Return a gpc object from one of the children.
getproperty(f::BlockGP, d::Symbol) = d === :gpc ? f.fs[1].gpc : getfield(f, d)
