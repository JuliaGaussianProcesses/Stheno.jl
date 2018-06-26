export JointGP

###################### Implementation of AbstractGaussianProcess interface #################

"""
    JointGP{Tfs<:AbstractVector{<:AbstractGP}} <: AbstractGaussianProcess

A collection of `AbstractGaussianProcess`es. Akin to the usual conception of a
Multi-Output Gaussian process.
"""
struct JointGP{Tfs<:AV{<:AbstractGP}} <: AbstractGaussianProcess
    fs::Tfs
end

mean(f::JointGP) = CatMean(mean.(f.fs))
kernel(f::JointGP) = CatKernel(kernel.(f.fs), kernel.(f.fs, permutedims(f.fs)))
kernel(f1::JointGP, f2::JointGP) = CatCrossKernel(kernel.(f1.fs, permutedims(f2.fs)))
kernel(f1::GP, f2::JointGP) = CatCrossKernel(kernel.(Ref(f1), permutedims(f2.fs)))
kernel(f1::JointGP, f2::GP) = CatCrossKernel(kernel.(f1.fs, Ref(f2)))



##################################### Syntactic Sugar ######################################

# Convenience methods for invoking `logpdf` and `rand` with multiple processes.
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}) = vec(rand(rng, fs, 1))
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}, N::Int) = rand(rng, JointGP(fs), N)
logpdf(fs::AV{<:AbstractGP}, ys::AV{<:AV{<:Real}}) = logpdf(JointGP(fs), BlockVector(ys))
