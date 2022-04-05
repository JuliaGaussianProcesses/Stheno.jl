# # The Convolutional Gaussian Process

using AbstractGPs
using KernelFunctions
using LinearAlgebra
using MLDatasets
using Random
using Stheno

using AbstractGPs: AbstractGP
using Stheno: DerivedGP

# Load MNIST training data set.
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
train_x_full, train_y_full = MNIST.traindata(Float32);

# Pull out the 1s and 2s.
ones_and_twos_indices = findall(y -> y == 1 || y == 2, train_y_full);
train_x = train_x_full[:, :, ones_and_twos_indices];
train_y = train_y_full[ones_and_twos_indices];

# Simple wrapper type representing a stack of greyscale images.
struct GreyScaleImageVector{T<:Real} <: AbstractVector{Matrix{T}}
    X::Array{T, 3}
end

Base.getindex(x::GreyScaleImageVector, n::Integer) = x.X[:, :, n]

Base.size(x::GreyScaleImageVector) = (size(x.X, 3), )

function extract_patches(x::GreyScaleImageVector)
    X = x.X
    return [
        ColVecs(reshape(getindex(X, p:p+2, q:q+2, :), :, size(X, 3))) for
        p in 1:size(X, 1)-2 for q in 1:size(X, 2)-2
    ]
end

# ## Specify the linear transformation

patch_convolve(g::AbstractGP) = DerivedGP((patch_convolve, g), g.gpc)

const patch_args = Tuple{typeof(patch_convolve), AbstractGP}

function AbstractGPs.mean((_, g)::patch_args, x::GreyScaleImageVector)
    return sum(map(xp -> mean(g, xp), extract_patches(x)))
end

function AbstractGPs.cov(
    (_, g)::patch_args, f′::AbstractGP, x::GreyScaleImageVector, x′::AbstractVector
)
    return sum(map(xp -> cov(g, xp, x′), extract_patches(x)))
end

function AbstractGPs.cov(
    f::AbstractGP, (_, g)::patch_args, x::AbstractVector, x′::GreyScaleImageVector
)
    return sum(map(xp′ -> cov(g, x, xp′), extract_patches(x′)))
end

function AbstractGPs.cov(
    (_, g)::patch_args, x::GreyScaleImageVector, x′::GreyScaleImageVector
)
    xps = extract_patches(x)
    xp′s = extract_patches(x′)
    return sum(map(xp′ -> sum(map(xp -> cov(g, xp, xp′), xps)), xp′s))
end

AbstractGPs.cov(args::patch_args, x::GreyScaleImageVector) = cov(args, x, x)

function AbstractGPs.var((_, g)::patch_args, x::GreyScaleImageVector)
    xps = extract_patches(x)
    return sum(map(xp′ -> sum(map(xp -> var(g, xp, xp′), xps)), xps))
end

# ## Specify a simple GPPP using this transformation.
function build_gp(θ)
    return @gppp let
        g = GP(θ.var * with_lengthscale(SEKernel(), θ.l))
        f = patch_convolve(g)
    end
end

# Check that things work.
x = GreyScaleImageVector(train_x[:, :, 1:10]);

f = build_gp((var=1, l=1))
mean(f, GPPPInput(:f, x))
cov(f, GPPPInput(:f, x), GPPPInput(:g, extract_patches(x)[1]))
cov(f, GPPPInput(:f, x), GPPPInput(:f, x))
var(f, GPPPInput(:f, x))

var(f, GPPPInput(:f, x)) ≈ diag(cov(f, GPPPInput(:f, x), GPPPInput(:f, x)))

# ## Apply pseudo-point approximation
M = 100;
z = GPPPInput(:g, ColVecs(randn(9, M)));
x = GPPPInput(:f, GreyScaleImageVector(train_x[:, :, 1:15]));

y = rand(f(x, 0.1))

cov(f, x, z)

elbo(VFE(f(z)), f(x, 0.1), y)
