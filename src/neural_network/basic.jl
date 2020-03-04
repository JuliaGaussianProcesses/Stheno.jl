export LinearLayer, ProductLayer, chain

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


# when writing ProductLayer, we don't use `prod`, because broadcasting problem will
# results in gradient evaluation problem.
struct ProductLayer <: AbstractModel
	step::Int
end
get_iparam(::ProductLayer) = Union{}[]
child(::ProductLayer) = ()
function (p::ProductLayer)(x)
	m, n = size(x)
	x1 = reshape(x, p.step, m÷p.step, n)
	res = .*([x1[i, :, :] for i in 1:p.step]...)
	return res
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

