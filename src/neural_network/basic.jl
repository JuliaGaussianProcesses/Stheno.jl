export LinearLayer, Product, chain

using Base: tail


softplus(x) = log(1+exp(x))
struct LinearLayer{T, MT<:AM{T}} <: AbstractModel
    W::MT
end
get_iparam(l::LinearLayer) = l.W
child(l::LinearLayer) = ()
LinearLayer(in_dim, out_dim) = LinearLayer(randn(out_dim, in_dim))
(lin::LinearLayer)(x) = softplus.(lin.W) * x

function Base.show(io::IO, layer::LinearLayer)
    print(io, "LinearLayer(", size(layer.W, 2), ", ", size(layer.W, 1), ")")
end


struct Product <: AbstractModel
	list::Tuple{Vararg{AV{Int}}}
    Product(Is...) = new(Is)
end
get_iparam(::Product) = Union{}[]
child(::Product) = ()
function (p::Product)(x)
	res = [prod(x[indices, :], dims=1) for indices in p.list]
	return vcat(res...)
end


struct Chain <: AbstractModel
	models::Tuple{Vararg{AbstractModel}}
	Chain(ms...) = new(ms)
end
get_iparam(::Chain) = Union{}[]
child(c::Chain) = c.models
applychain(::Tuple{}, x) = x
applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))
(c::Chain)(x) = applychain(c.models, x)
chain(ms...) = Chain(ms...)

