export AbstractModel, parameters, parameter_eltype, dispatch!, extract_gradient


"""
Here I introduce an `AbstractModel` type, acting as the root type of all types that contain
learnable parameters, such as `AbstractGP`, `Kernel` & `Meanfunction`. This gives
our model a tree structure, and facilitate for collecting and redistributing parameters.

Here is an example of how our GP model now looks like:

							     GP
							    /  \
							   /    \
				                 ConstantMean   Scaled
					             (c)         (σ)
						                  |
							       Stretched
								 (l)
							          |
							         EQ()

Parameters for this model are `c`, `σ` & `l`, we can use:
```julia
θ = parameters(GP)
```
to extract all parameters contained in the model, and
```julia
dispatch!(GP, θ₁)
```
to redistribute the updated parameters ( maybe returned from some optimizer ) back to the model.

Enabling parameter collecting and dispatching features leave Stheno's current type implementation and
APIs unchanged, one only need to subtype the current types to `AbstractModel` type and add two interfaces
`get_iparam` & `child` for each one, for example:
```julia
struct EQ <: AbstractModel end
get_iparam(::EQ) = Union{}[]
child(::EQ) = ()
```
"""


const AVM = AbstractVecOrMat

abstract type AbstractModel end

# Return parameters contained inside a model
get_iparam(m::AbstractModel) = throw(UndefVarError("get_iparam method not defined for $m"))
# Return model that contained in another model, e.g. `Stretched` contains kernel
child(m::AbstractModel) = throw(UndefVarError("child method not defined for $m"))


# parameter_eltype will return the type of each paramters inside a model, for those types that 
# are not subtype of `AbstractModel`, and cases where a model contains no parameters, e.g. EQ kernel, 
# it will return `Union{}`.
parameter_eltype(::Any) = Union{}
function parameter_eltype(x::AbstractModel)
    T = eltype(get_iparam(x))
    for each in child(x)
        T = promote_type(T, parameter_eltype(each))
    end
    return T
end


# Extract all parameters of a model to a 1D array
parameters(x::AbstractModel) = parameters!(parameter_eltype(x)[], x)
parameters!(out, ::Any) = out
function parameters!(out, x::AbstractModel)
    append!(out, get_iparam(x))
    for x_child in child(x)
        parameters!(out, x_child)
    end
    return out
end


# Return number of parameters contained inside a model
get_nparameter(x::AbstractModel) = length(parameters(x))


# dispatch! allows us to update parameters inside a model, it accept a model and a 1D
# array, it will assign values inside the array to the corresponding parameter of the model. 
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


# Zygote is able to compute gradient w.r.t a parametrized type, for example:
# ```
# struct Linear
#   W
# 	b
# end
#
# (l::Linear)(x) = l.W * x .+ l.b
# model = Linear(rand(2, 5), rand(2))
# dmodel = gradient(model -> sum(model(x)), model)[1]
# ```
# the results is a `NamedTuple`. `extract_gradient` function is used
# to extract the value of those gradients to a 1D array.
extract_gradient(k::AbstractModel, G::NamedTuple) = extract_gradient!(parameter_eltype(k)[], G)
function extract_gradient!(out, G::NamedTuple)
    for (_, val) in pairs(G)
        if val isa AVM
	    append!(out, val)
	elseif val isa NamedTuple
	    extract_gradient!(out, val)
	elseif val isa Tuple
	    for each in val
		if each isa NamedTuple
		    extract_gradient!(out, each)
		end
	    end
	end
    end
    return out
end



