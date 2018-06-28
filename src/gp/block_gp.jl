export BlockGP

###################### Implementation of AbstractGaussianProcess interface #################

"""
    BlockGP{Tfs<:AbstractVector{<:AbstractGP}} <: AbstractGaussianProcess

A collection of `AbstractGaussianProcess`es. Akin to the usual conception of a
Multi-Output Gaussian process.
"""
struct BlockGP{Tfs<:AV{<:AbstractGP}} <: AbstractGaussianProcess
    fs::Tfs
end
BlockGP(fs::Tuple) = BlockGP([fs...])

mean(f::BlockGP) = BlockMean(mean.(f.fs))
kernel(f::BlockGP) = BlockKernel(kernel.(f.fs), kernel.(f.fs, permutedims(f.fs)))
kernel(f1::BlockGP, f2::BlockGP) = BlockCrossKernel(kernel.(f1.fs, permutedims(f2.fs)))
kernel(f1::GP, f2::BlockGP) = BlockCrossKernel(kernel.(Ref(f1), permutedims(f2.fs)))
kernel(f1::BlockGP, f2::GP) = BlockCrossKernel(kernel.(f1.fs, Ref(f2)))



##################################### Syntactic Sugar ######################################

# Convenience methods for invoking `logpdf` and `rand` with multiple processes.
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}) = vec(rand(rng, fs, 1))
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}, N::Int) = rand(rng, BlockGP(fs), N)
logpdf(fs::AV{<:AbstractGP}, ys::AV{<:AV{<:Real}}) = logpdf(BlockGP(fs), BlockVector(ys))
