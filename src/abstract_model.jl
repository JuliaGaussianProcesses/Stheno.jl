export AbstractModel, parameters, parameter_eltype, dispatch!, extract_gradient

const AVM = AbstractVecOrMat

abstract type AbstractModel end

function get_iparam(::AbstractModel) end
function child(::AbstractModel) end
parameters(x::AbstractModel) = parameters!(parameter_eltype(x)[], x)
function parameters!(out, x::AbstractModel)
    append!(out, get_iparam(x))
		for x_child in child(x)
        parameters!(out, x_child)
    end
    return out
end
function parameter_eltype(x::AbstractModel)
	  T = eltype(get_iparam(x))
    for each in child(x)
        T = promote_type(T, parameter_eltype(each))
    end
    return T
end
get_nparameter(x::AbstractModel) = length(parameters(x))

function dispatch!(k::AbstractModel, v::AV)
	  nθ_k = get_nparameter(k)
		nθ_k == length(v) || throw(DimensionMismatch("expect $(nθ_k) parameters, got $(length(v))"))
    θ = get_iparam(k)
    copyto!(θ, 1, v, 1, length(θ))
    loc = 1 + length(θ)
    for k′ in child(k)
			  nθ_k′ = get_nparameter(k′)
			  dispatch!(k′, v[loc:loc+nθ_k′-1])
        loc += nθ_k′
    end
    return k
end
extract_gradient(k::AbstractModel, G::NamedTuple) = extract_gradient!(parameter_eltype(k)[], G)
function extract_gradient!(out, G::NamedTuple)
	for (_, val) in pairs(G)
		if val isa AVM
			append!(out, val)
		elseif val isa NamedTuple
			extract_gradient!(out, val)
		elseif val isa Tuple && eltype(val) == NamedTuple
			foreach(x->extract_gradient!(out, x), val)
		end
	end
	return out
end



