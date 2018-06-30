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

##################################### Syntactic Sugar ######################################

# Convenience methods for invoking `logpdf` and `rand` with multiple processes.
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}) = vec(rand(rng, fs, 1))
rand(rng::AbstractRNG, fs::AV{<:AbstractGP}, N::Int) = rand(rng, BlockGP(fs), N).blocks
function rand(rng::AbstractRNG, f::BlockGP, N::Int)
    M = BlockArray(uninitialized_blocks, AbstractMatrix{Float64}, length.(f.fs), [N])
    μ = mean_vec(f)
    for b in eachindex(f.fs)
        setblock!(M, getblock(μ, b) * ones(1, N), b, 1)
    end
    return M + chol(cov(f))' * BlockMatrix(randn.(rng, length.(f.fs), N))
end

logpdf(fs::AV{<:AbstractGP}, ys::AV{<:AV{<:Real}}) = logpdf(BlockGP(fs), BlockVector(ys))

# function rand(rng::AbstractRNG, f::AV{<:GP}, X::AV{<:AVM}, N::Int)
#     ϵ = BlockMatrix(randn.(rng, nobs.(X), N))
#     y = mean(f, X) .+ chol(cov(f, X))' * ϵ
#     ends = cumsum(nobs.(X))
#     starts = ends .- nobs.(X) .+ 1
#     return [y[starts[n]:ends[n], :] for n in eachindex(starts)]
# end
# rand(rng::AbstractRNG, f::AV{<:GP}, X::AV{<:AVM}) = vec.(rand(rng, f, X, 1))
# rand(rng::AbstractRNG, f::GP, X::AVM, N::Int) = rand(rng, [f], [X], N)[1]
# rand(rng::AbstractRNG, f::GP, X::AVM) = rand(rng, [f], [X])[1]
