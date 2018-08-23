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

deconstruct(f::BlockGP) = (f.fs...,)

# Return a gpc object from one of the children.
function getproperty(f::BlockGP, d::Symbol)
    if d === :gpc
        return f.fs[1].gpc
    else
        return getfield(f, d)
    end
end



##################################### Syntactic Sugar ######################################

# Specialised implementation of `rand` for `BlockGP`s.
function rand(rng::AbstractRNG, f::BlockGP, N::Int)
    M = BlockArray(undef_blocks, AbstractMatrix{Float64}, length.(f.fs), [N])
    μ = mean_vec(f)
    for b in eachindex(f.fs)
        setblock!(M, getblock(μ, b) * ones(1, N), b, 1)
    end
    return M + chol(cov(f))' * BlockMatrix(randn.(Ref(rng), length.(f.fs), N))
end
rand(f::BlockGP, N::Int) = rand(Base.Random.GLOBAL_RNG, f, N)

function rand(rng::AbstractRNG, f::BlockGP)
    return mean_vec(f) + chol(cov(f))' * BlockVector(randn.(Ref(rng), length.(f.fs)))
end
rand(f::BlockGP) = rand(Base.Random.GLOBAL_RNG, f)

# Convenience methods for invoking `logpdf` and `rand` with multiple processes.
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}) = vec.(rand(rng, fs, 1))
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}, N::Int) = rand(rng, BlockGP(fs), N).blocks
rand(fs::AV{<:AbstractGP}, N::Int) = rand(Random.GLOBAL_RNG, fs, N)
rand(fs::AV{<:AbstractGP}) = vec.(rand(Random.GLOBAL_RNG, fs))
logpdf(fs::AV{<:AbstractGP}, ys::AV{<:AV{<:Real}}) = logpdf(BlockGP(fs), BlockVector(ys))
