export Primitive, NeuralKernelNetwork


# Primitive layer, mainly act as a container to hold basic kernels for the neural kernel network
struct Primitive{T} 
    kernels::T
    Primitive(ks...) = new{typeof(ks)}(ks)
end
child(p::Primitive) = p.kernels

# flatten k kernel matrices of size Mk×Nk to 1×(Mk_Nk), and concatenate these array into a k×(Mk_Nk) 2d array
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
# nerual network architecture to perform kernel composition. It should work like a normal `Stheno` kernel.
struct NeuralKernelNetwork{PT<:Primitive, NNT<:Union{LinearLayer, ProductLayer, Chain}} <: Kernel
    primitives::PT
    nn::NNT
end
child(nkn::NeuralKernelNetwork) = (nkn.primitives, nkn.nn)

# use this function to reshape the 1d array back to kernel matrix
_rebuild_kernel(x, n, m) = reshape(x, n, m)
# the result of `ew` function should be a 1d array, however, the result of Flux's neural network is a 2d array,
# therefore, we reshape it to 1d
_rebuild_diag(x) = reshape(x, :)

ew(nkn::NeuralKernelNetwork, x) = _rebuild_diag(nkn.nn(ew(nkn.primitives, x)))
pw(nkn::NeuralKernelNetwork, x) = _rebuild_kernel(nkn.nn(pw(nkn.primitives, x)), length(x), length(x))

ew(nkn::NeuralKernelNetwork, x, x′) = _rebuild_diag(nkn.nn(ew(nkn.primitives, x, x′)))
pw(nkn::NeuralKernelNetwork, x, x′) = _rebuild_kernel(nkn.nn(pw(nkn.primitives, x, x′)), length(x), length(x′))

function Base.show(io::IO, kernel::NeuralKernelNetwork)
    print(io, "NeuralKernelNetwork(")
    join(io, [kernel.primitives, kernel.nn], ", ")
    print(io, ")")
end


