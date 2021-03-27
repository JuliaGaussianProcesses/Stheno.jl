"""
    

"""
struct GaussianProcessProbabilisticProgramme{Tfs<:Dict} <: AbstractGP
    fs::Tfs
    gpc::GPC
end

const GPPP = GaussianProcessProbabilisticProgramme

"""


"""
struct GPPPInput{Tp, T, Tx<:AbstractVector{T}} <: AbstractVector{Tuple{Tp, T}}
    p::Tp
    x::Tx
end

Base.size(x::GPPPInput) = (length(x.x), )

Base.getindex(x::GPPPInput, idx) = map(x_ -> (x.p, x_), x.x[idx])



# As a first-pass, I'm letting the backend involve integer arithmetic, an planning to build
# helper functionality on top.

AbstractGPs.mean(f::GPPP, x::GPPPInput) = mean(f.fs[x.p], x.x)

AbstractGPs.cov(f::GPPP, x::GPPPInput) = cov(f.fs[x.p], x.x)

AbstractGPs.cov_diag(f::GPPP, x::GPPPInput) = cov_diag(f.fs[x.p], x.x)

function AbstractGPs.cov(f::GPPP, x::GPPPInput, x′::GPPPInput)
    return cov(f.fs[x.p], f.fs[x′.p], x.x, x′.x)
end

function AbstractGPs.cov_diag(f::GPPP, x::GPPPInput, x′::GPPPInput)
    return cov_diag(f.fs[x.p], f.fs[x′.p], x.x, x′.x)
end

AbstractGPs.mean_and_cov(f::GPPP, x::GPPPInput) = mean_and_cov(f.fs[x.p], x.x)

AbstractGPs.mean_and_cov_diag(f::GPPP, x::GPPPInput) = mean_and_cov_diag(f.fs[x.p], x.x)


using MacroTools: postwalk, splitdef, combinedef, @capture, prewalk

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
