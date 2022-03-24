# # The Convolutional Gaussian Process

using AbstractGPs
using KernelFunctions
using LinearAlgebra
using MLDatasets
using Random
using Stheno
using Zygote

using AbstractGPs: AbstractGP
using Stheno: DerivedGP

# Load MNIST training data set.
train_x_full, train_y_full = MNIST.traindata(Float32);

# Pull out the 1s and 2s.
ones_and_twos_indices = findall(y -> y == 1 || y == 2, train_y_full);
train_x = train_x_full[:, :, ones_and_twos_indices];
train_y = train_y_full[ones_and_twos_indices];

function extract_patches(X::AbstractArray{<:Real, 3})
    patches_vec = map(1:size(X, 3)) do n
        reduce(
            hcat,
            [vec(getindex(X, p:p+2, q:q+2, n)) for p in 1:size(X, 1)-2, q in 1:size(X, 2)-2],
        )
    end
    return reduce(hcat, patches_vec)
end

# Simple wrapper type representing a stack of greyscale images.
struct GreyScaleImageVector{T<:Real} <: AbstractVector{Matrix{T}}
    X::Array{T, 3}
    patches::Matrix{T}
end

GreyScaleImageVector(X::Array{<:Real, 3}) = GreyScaleImageVector(X, extract_patches(X))

Base.getindex(x::GreyScaleImageVector, n::Integer) = x.X[:, :, n]

Base.size(x::GreyScaleImageVector) = (size(x.X, 3), )

# ## Specify the linear transformation

patch_convolve(g::AbstractGP) = DerivedGP((patch_convolve, g), g.gpc)

const patch_args = Tuple{typeof(patch_convolve), AbstractGP}

function AbstractGPs.mean((_, g)::patch_args, x::GreyScaleImageVector)
    m_patches = mean(g, ColVecs(x.patches))
    M = reshape(m_patches, :, length(x))
    return vec(sum(M; dims=1))
end

function AbstractGPs.cov(
    (_, g)::patch_args, f′::AbstractGP, x::GreyScaleImageVector, x′::AbstractVector
)
    k_patches = cov(g, f′, ColVecs(x.patches), x′)
    K = reshape(k_patches, :, length(x), length(x′))
    return dropdims(sum(K; dims=1), dims=1)
end

function AbstractGPs.cov(
    (_, g)::patch_args, x::GreyScaleImageVector, x′::GreyScaleImageVector
)
    k_patches = cov(g, ColVecs(x.patches), ColVecs(x′.patches))
    K = 
end

x = GreyScaleImageVector(train_x[:, :, 1:10]);
f = @gppp let
    g = GP(SEKernel())
    f = patch_convolve(g)
end

mean(f, GPPPInput(:f, x))
cov(f, GPPPInput(:f, x), GPPPInput(:g, ColVecs(x.patches[:, 1:7])))
