"""
    GaussianProcessProbabilisticProgramme(fs, gpc)

Collects a group of related GPs together and interprets them as a single GP.
This type isn't part of the user-facing API -- the `@gppp` macro should be used to
construct a `GaussianProcessProbabilisticProgramme`.
"""
struct GaussianProcessProbabilisticProgramme{Tfs} <: AbstractGP
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

function extract_components(f::GPPP, x::AbstractVector{<:Tuple{T, V}} where {T, V})
    symbols = first.(x)
    features = last.(x)

    # Find the indices of `x` associated with each unique symbol.
    unique_symbols = unique(symbols)
    block_inds = map(s -> findall(t -> t == s, symbols), unique_symbols)

    # Construct a BlockData comprising `GPPPInput`s, and extract_components from that.
    blocks = map((s, inds) -> GPPPInput(s, features[inds]), unique_symbols, block_inds)
    return extract_components(f, BlockData(blocks))
end

function mean(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return mean(fs, vs)
end

function cov(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return cov(fs, vs)
end

function var(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return var(fs, vs)
end

function cov(f::GPPP, x::AbstractVector, x′::AbstractVector)
    fs_x, vs_x = extract_components(f, x)
    fs_x′, vs_x′ = extract_components(f, x′)
    return cov(fs_x, fs_x′, vs_x, vs_x′)
end

function var(f::GPPP, x::AbstractVector, x′::AbstractVector)
    fs_x, vs_x = extract_components(f, x)
    fs_x′, vs_x′ = extract_components(f, x′)
    return var(fs_x, fs_x′, vs_x, vs_x′)
end

function mean_and_cov(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return mean_and_cov(fs, vs)
end

function mean_and_var(f::GPPP, x::AbstractVector)
    fs, vs = extract_components(f, x)
    return mean_and_var(fs, vs)
end



"""
    Base.split(x::BlockData, Y::AbstractVecOrMat)

Convenience functionality to make working with the output of operations on GPPPs easier.
Splits up the rows of `Y` according to the sizes of the data in `x`.

```jldoctest
julia> Y = vcat(randn(5, 3), randn(4, 3));

julia> x = BlockData(randn(5), randn(4));

julia> Y1, Y2 = split(x, Y);

julia> Y1 == Y[1:5, :]
true

julia> Y2 == Y[6:end, :]
true
```

Works with any `BlockData`, so blocks can e.g. be `GPPPInput`s. This is particularly helpful
for working with the output from `rand` and `marginals` from a `GPPP` indexed at
`BlockData`. For example
```julia
f = @gppp let
    f1 = GP(SEKernel())
    f2 = GP(Matern52Kernel())
    f3 = f1 + f2
end

x = BlockData(GPPPInput(:f2, randn(3)), GPPPInput(:f3, randn(4)))
y = rand(f(x))
y2, y3 = split(x, y)
```

Functionality also works with any `AbstractVector`.
"""
function Base.split(x::BlockData, Y::AbstractMatrix)
    length(x) == size(Y, 1) || throw(error("Expected length(x) == size(Y, 1)"))
    return map(idx->Y[idx, :], _get_indices(x))
end

function Base.split(x::BlockData, y::AbstractVector)
    length(x) == length(y) || throw(error("Expected length(x) == length(y)"))
    return map(idx->y[idx], _get_indices(x))
end

function _get_indices(x::BlockData)
    sz = cumsum(map(length, x.X))
    return [sz[n] - length(x.X[n]) + 1:sz[n] for n in eachindex(x.X)]
end
ChainRulesCore.@non_differentiable _get_indices(::Any)



"""
    @gppp(model_expression)

Construct a `GaussianProcessProbabilisticProgramme` (`GPPP`) from a code snippet.

```jldoctest
f = @gppp let
    f1 = GP(SEKernel())
    f2 = GP(Matern52Kernel())
    f3 = f1 + f2
end

x = GPPPInput(:f3, randn(5))

y = rand(f(x, 0.1))

logpdf(f(x, 0.1), y) ≈ elbo(f(x, 0.1), y, f(x, 1e-9))

# output

true
```
"""
macro gppp(let_block::Expr)
    # Consult the `Internals` section of the docs for info about what this macro is doing.

    # Ensure that we're dealing with a let block.
    let_block.head == :let || throw(error("gppp needs a let block."))

    # Pull out model definition from the block.
    model_expr = let_block.args[2]
    model_expr.head == :block || throw(error("Expected a block."))

    # Pull out the names of all of the variables.
    lines_declaring_variables = filter(x -> x isa Expr && x.head == :(=), model_expr.args)
    variable_names = map(x -> x.args[1], lines_declaring_variables)

    # Construct expression which specifies mappings between symbolic names and GPs.
    # The resulting expression is of the form (f1 = f1, f2 = f2).
    var_mapping = Expr(
        :tuple,
        map(variable_names) do var_name
            Expr(:(=), var_name, var_name)
        end...,
    )

    gpc_sym = gensym("gpc")

    # Construct an expression which wraps the GPs, and returns a GPPP.
    wrapped_model = Expr(:block,
        :($gpc_sym = Stheno.GPC()),
        postwalk(
            x->@capture(x, GP(xs__)) ? :(Stheno.wrap(GP($(xs...)), $gpc_sym)) : x,
            model_expr,
        ).args...,
        :(Stheno.GPPP($var_mapping, $gpc_sym)),
    )
    return esc(wrapped_model)
end
