export LinearLayer, product, Primitive, NeuralKernelNetwork

using .Flux
using .Flux: softplus, @functor



# Linear layer, perform linear transformation to input array
# x₁ = W * x₀
struct LinearLayer{T, MT<:AbstractArray{T}}
    W::MT
end
@functor LinearLayer
LinearLayer(in_dim, out_dim) = LinearLayer(randn(out_dim, in_dim))
(lin::LinearLayer)(x) = softplus.(lin.W) * x

function Base.show(io::IO, layer::LinearLayer)
	print(io, "LinearLayer(", size(layer.W, 2), ", ", size(layer.W, 1), ")")
end


# Product function, given an 2d array whose size is M×N, product layer will
# multiply every m neighboring rows of the array elementwisely to obtain
# an new array of size (M÷m)×N
function product(x, step=2)
	m, n = size(x)
	m%step == 0 || error("the first dimension of inputs must be multiple of step")
	new_x = reshape(x, step, m÷step, n)
	.*([new_x[i, :, :] for i in 1:step]...)
end


# Primitive layer, mainly act as a container to hold basic kernels for the neural kernel network
struct Primitive{T}
    kernels::T
    Primitive(ks...) = new{typeof(ks)}(ks)
end
@functor Primitive

# flatten k kernel matrices of size Mk×Nk, and concatenate these 1d array into a k×(Mk*Nk) 2d array
_cat_kernel_array(x) = vcat([reshape(x[i], 1, :) for i in 1:length(x)]...)

# NOTE, though we implement `ew` & `pw` function for Primitive, it isn't a subtype of Kernel type,
# I do this because it will facilitate writing NeuralKernelNetwork
ew(p::Primitive, x) = _cat_kernel_array(map(k->ew(k, x), p.kernels))
pw(p::Primitive, x) = _cat_kernel_array(map(k->pw(k, x), p.kernels))

ew(p::Primitive, x, x′) = _cat_kernel_array(map(k->ew(k, x, x′), p.kernels))
pw(p::Primitive, x, x′) = _cat_kernel_array(map(k->pw(k, x, x′), p.kernels))

function Base.show(io::IO, layer::Primitive)
	print(io, "Primitive(")
	join(io, layer.kernels, ", ")
	print(io, ")")
end


# Neural Kernel Network, since kernel space ( stationary kernel ) is closed under linear combination 
# ( with positive coefficient ) and element-wise multiplication, we can use a neural network like structure
# to build composite kernels. This type contains a `Primitive` layer which holds basic kerenls and a specialised
# nerual network architecture to perform kernel composition. It should function like a normal `Stheno` kernel.
struct NeuralKernelNetwork{PT, CT} <: Kernel
    player::PT
    chain::CT
end
@functor NeuralKernelNetwork

# use this function to reshape the 1d array back to kernel matrix
_rebuild_kernel(x, n, m) = reshape(x, n, m)
_rebuild_diag(x) = reshape(x, :)

ew(nkn::NeuralKernelNetwork, x) = _rebuild_diag(nkn.chain(ew(nkn.player, x)))
pw(nkn::NeuralKernelNetwork, x) = _rebuild_kernel(nkn.chain(pw(nkn.player, x)), length(x), length(x))

ew(nkn::NeuralKernelNetwork, x, x′) = _rebuild_diag(nkn.chain(ew(nkn.player, x, x′)))
pw(nkn::NeuralKernelNetwork, x, x′) = _rebuild_kernel(nkn.chain(pw(nkn.player, x, x′)), length(x), length(x′))

function Base.show(io::IO, kernel::NeuralKernelNetwork)
	print(io, "NeuralKernelNetwork(")
	join(io, [kernel.player, kernel.chain], ", ")
	print(io, ")")
end


