"""
    GaussianProcessProbabilisticProgramme(fs, gpc)

Collects a group of related GPs together and interprets them as a single GP.
This type isn't part of the user-facing API -- the `@gppp` macro should be used to
construct a `GaussianProcessProbabilisticProgramme`.
"""
struct GaussianProcessProbabilisticProgramme{Tfs<:Dict} <: AbstractGP
    fs::Tfs
    gpc::GPC
end

const GPPP = GaussianProcessProbabilisticProgramme

"""
    GPPPInput(p, x::AbstractVector)

An collection of inputs for a GPPP.
`p` indicates which process the vector `x` should be extracted from.
The required type of `p` is determined by the type of the keys in the `GPPP` indexed.

```jldoctest
julia> x = [1.0, 1.5, 0.3];

julia> v = GPPPInput(:a, x)
3-element GPPPInput{Symbol, Float64, Vector{Float64}}:
 (:a, 1.0)
 (:a, 1.5)
 (:a, 0.3)

julia> v isa AbstractVector{Tuple{Symbol, Float64}}
true

julia> v == map(x_ -> (:a, x_), x)
true
```
"""
struct GPPPInput{Tp, T, Tx<:AbstractVector{T}} <: AbstractVector{Tuple{Tp, T}}
    p::Tp
    x::Tx
end

Base.size(x::GPPPInput) = (length(x.x), )

Base.getindex(x::GPPPInput, idx) = map(x_ -> (x.p, x_), x.x[idx])



#
# Implementation of the AbstractGPs API for the inputs types which are valid for a GPPP.
# See `AbstractGPs.jl` for details.
#

extract_components(f::GPPP, x::GPPPInput) = f.fs[x.p], x.x

function extract_components(f::GPPP, x::BlockData)
    fs = map(v -> f.fs[v.p], x.X)
    vs = map(v -> v.x, x.X)
    return cross(fs), BlockData(vs)
end

function AbstractGPs.mean(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return mean(fs, vs)
end

function AbstractGPs.cov(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return cov(fs, vs)
end

function AbstractGPs.cov_diag(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return cov_diag(fs, vs)
end

function AbstractGPs.cov(f::GPPP, x::AbstractVector, x′::AbstractVector)
    fs_x, vs_x = extract_components(f, x)
    fs_x′, vs_x′ = extract_components(f, x′)
    return cov(fs_x, fs_x′, vs_x, vs_x′)
end

function AbstractGPs.cov_diag(f::GPPP, x::AbstractVector, x′::AbstractVector)
    fs_x, vs_x = extract_components(f, x)
    fs_x′, vs_x′ = extract_components(f, x′)
    return cov_diag(fs_x, fs_x′, vs_x, vs_x′)
end

function AbstractGPs.mean_and_cov(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return mean_and_cov(fs, vs)
end

function AbstractGPs.mean_and_cov_diag(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return mean_and_cov_diag(fs, vs)
end



"""
    @gppp(model_expression)

Construct a `GaussianProcessProbabilisticProgramme` (`GPPP`) from a code snippet.
"""
macro gppp(let_block::Expr)

    # Ensure that we're dealing with a let block.
    let_block.head == :let || throw(error("gppp needs a let block."))

    # Pull out model definition from the block.
    model_expr = let_block.args[2]
    model_expr.head == :block || throw(error("Expected a block."))

    # Pull out the names of all of the variables.
    lines_declaring_variables = filter(x -> x isa Expr && x.head == :(=), model_expr.args)
    variable_names = map(x -> x.args[1], lines_declaring_variables)

    # Construct expression which specifies mappings between symbolic names and GPs.
    # The resulting expression is of the form [:f1 => f1, :f2 => f2].
    var_mapping = Expr(
        :vect,
        map(variable_names) do var_name
            Expr(:call, :(=>), QuoteNode(var_name), var_name)
        end...,
    )

    # Construct an expression which wraps the GPs, and returns a GPPP.
    wrapped_model = Expr(:block,
        :(gpc = Stheno.GPC()),
        postwalk(
            x->@capture(x, GP(xs__)) ? :(Stheno.wrap(GP($(xs...)), gpc)) : x,
            model_expr,
        ).args...,
        :(GPPP(Dict($var_mapping), gpc)),
    )
    return esc(wrapped_model)
end
