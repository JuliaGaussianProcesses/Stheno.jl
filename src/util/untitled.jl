using Revise
using LinearAlgebra
using LinearAlgebra: RealHermSymComplexHerm, copytri!
import LinearAlgebra: cholesky, UpperTriangular, logdet, \
const chol_type = Union{StridedMatrix, RealHermSymComplexHerm{<:Real, <:StridedMatrix}}

using Flux.Tracker
using Flux.Tracker: tracker, Call, track, TrackedVecOrMat, Tracked, TrackedArray,
    TrackedReal

# This probably isn't optimal, but probably isn't horrific.
function chol_sensitivity(
    U::AbstractMatrix{T},
    Ū::AbstractMatrix{T},
    Σ::AbstractMatrix{T},
) where T<:Real
    Σ̄ = Ū * UpperTriangular(U)'
    Σ̄ = copytri!(Σ̄, 'U')
    Σ̄ = ldiv!(UpperTriangular(U), Σ̄)
    BLAS.trsm!('R', 'U', 'T', 'N', one(T), U, Σ̄)
    @inbounds for n in diagind(Σ̄)
        Σ̄[n] *= 0.5
    end
    return UpperTriangular(Σ̄)
end

# This is a fairly incomplete implementation of the Cholesky that just about works in the
# most basic situation. It should, in principle, be straightforward to generalise.
function cholesky(Σ::TrackedMatrix{<:Real, <:chol_type})

    # Compute the Cholesky factorisation as per usual.
    C_raw = cholesky(Tracker.data(Σ))
    U = C_raw.factors

    # Compute reverse-pass function.
    C̄ = Ū->(chol_sensitivity(Tracker.data(U), Tracker.data(Ū), Tracker.data(Σ)),)

    # Track factors from Cholesky factorisation. Nothing else needs to be touched.
    tracked_factors = track(Call(C̄, (tracker(Σ),)), U)

    # Pack everything into a Cholesky.
    return Cholesky{eltype(Σ), typeof(tracked_factors)}(tracked_factors, C_raw.uplo, C_raw.info)
end

# This is also a hack, but does the job nicely.
logdet(C::Cholesky{<:Real, <:TrackedMatrix}) = 2 * sum(log, C.factors[diagind(C.factors)])

# Sensible implementation of UpperTriangular.
using Flux.Tracker: @grad

# U = UpperTriangular(X)
UpperTriangular(X::TrackedMatrix) = Tracker.track(UpperTriangular, X)
@grad function UpperTriangular(X)
    return UpperTriangular(Tracker.data(X)), function (Ū)
        return (UpperTriangular(copy(Ū)),)
    end
end

# L = LowerTriangular(X)
LowerTriangular(X::TrackedMatrix) = Tracker.track(LowerTriangular, X)
@grad function LowerTriangular(X)
    return LowerTriangular(Tracker.data(X)), function(L̄)
        return (LowerTriangular(copy(L̄)),)
    end
end

# Y = A \ B
\(A::TrackedMatrix, B::TrackedVecOrMat) = Tracker.track(\, A, B)
\(A::AbstractMatrix, B::TrackedVecOrMat) = Tracker.track(\, A, B)
\(A::TrackedMatrix, B::AbstractVecOrMat) = Tracker.track(\, A, B)
@grad function \(A::AbstractMatrix, B::AbstractVector)
    Y = Tracker.data(A) \ Tracker.data(B)
    return Y, function(Ȳ)
        B̄ = A' \ Ȳ
        return (-B̄ * Y', B̄)
    end
end

# Y = A / B
/(A::TrackedVecOrMat, B::TrackedMatrix) = Tracker.track(/, A, B)
/(A::AbstractVecOrMat, B::TrackedMatrix) = Tracker.track(/, A, B)
/(A::TrackedVecOrMat, B::AbstractMatrix) = Tracker.track(/, A, B)
@grad function /(A::AbstractVecOrMat, B::AbstractMatrix)
    Y = Tracker.data(A) / Tracker.data(B)
    return Y, function(Ȳ)
        Ā = Ȳ / B'
        return (Ā, -Y' * Ā)
    end
end


N = 4
B = randn(N, N)
x = randn(N)
A = B' * B
C = cholesky(Symmetric(A))

foo(A) = logdet(cholesky(A))
bar(A) = sum(abs2, cholesky(A).U \ x)


using Flux.Tracker: forward
l, Ā = forward(foo, A)
Ā(1.0)[1]

l, Ā = forward(bar, A)
Ā(1.0)[1]

using Test




using ForwardDiff: gradient
gradient(foo, A)
gradient(bar, A)



using Stheno
using Stheno: @model

@model function gp(σ)
    return σ * GP(EQ())
end

# Generate toy data.
f = gp(1.0)
x = randn(2)
y = rand(f(x))


loss(σ) = -logpdf(gp(σ)(x), y)

l, σ̄ = forward(loss, 1.0)
∂σ = σ̄(1.0)[1]




"""
    Op{Tf, Tvalue, Targs}

The totality of a call to a (pure) primtive function `f` at `args` and `kwargs`,
producing `value`.
"""
struct Op{Tf, Tvalue, Targs}
    f::Tf
    value::Tvalue
    args::Targs
    function Op(f::Tf, args...) where Tf
        value = f(args...)
        return new{Tf, typeof(value), typeof(args)}(f, value, args)
    end
    function Op(value::T) where T
        return new{Nothing, T, Nothing}(nothing, value, nothing)
    end
end

# Alias for distinguishing between leaves and branches.
const Leaf = Op{Nothing}
is_leaf(::Leaf) = true
is_leaf(::Op) = false

value(op::Op) = op.value
#
# function show(io::IO, op::Op)
#     print(io, "(Op  ) $(op.f)")
# end
# function show(io::IO, mime::MIME"text/plain", op::Op)
#     println("Op where")
#     println("f = $(op.f)")
#     println("y = $(value(op))")
#     println("args = $(op.args)")
# end
#
# function show(io::IO, op::Leaf)
#     print(io, "(Leaf) $(typeof(op.value))")
# end

# struct TapePair
#     op::Op
#     positions::Tuple{Vararg{Int}}
# end
# operation(pair::TapePair) = pair.op
# positions(pair::TapePair) = pair.positions

const Tape = Vector{Any}

# function show(io::IO, mime::MIME"text/plain", tape::Tape)
#     if length(tape) == 0
#         print("0-element Tape")
#     else
#         println("$(length(tape))-element Tape:")
#         for (n, pair) in enumerate(tape)
#             if operation(pair) isa Leaf
#                 str = " %$n = $(operation(pair))"
#             else
#                 args = ["%" .* string.(positions(pair)) .* ", "...]
#                 for n in eachindex(args)
#                     if positions(pair)[n] == -1
#                         args[n] = string(typeof(operation(pair).args[n])) * ", "
#                     end
#                 end
#                 args[end] = args[end][1:end-2]
#                 str = " %$n = $(operation(pair))($(args...))"
#             end
#             (n == length(tape) ? print : println)(str)
#         end
#     end
# end

# A sprinkling of contextual execution.
using Cassette
using Cassette: @context, overdub, OverdubInstead, enabletagging
import Cassette: execute
@context FluxCtx

# Define what is tracked and what isn't.
is_tracked(::Any) = false
is_tracked(::TrackedArray) = true
is_tracked(::TrackedReal) = true

untrack(x::Union{TrackedReal, TrackedArray}) = x.data
untrack(x) = x

function Cassette.prehook(ctx::FluxCtx, ::typeof(track), f, args...; kwargs...)
    println("Inside general tracking prehook")
    println(f)
    println(args)
end
function Cassette.prehook(ctx::FluxCtx, ::typeof(track), call::Call, args...)
    println("Inside Call-specific prehook")
    push!(ctx.metadata, call.func)
end
# function Cassette.prehook(ctx::FluxCtx, ::typeof(sum), f, args...; kwargs...)

#     println("Inside sum tracking prehook")
# end
using Flux.Tracker: ∇broadcast
function Cassette.prehook(ctx::FluxCtx, ::typeof(∇broadcast), f, args...)
    println("Inside broadcast prehook")
end

# """
#     execute(ctx::FluxCtx, ::typeof(track), f, args...; kwargs...)

# Intercept calls to Flux.track.
# """
# function execute(ctx::FluxCtx, ::typeof(track), f, args...; kwargs...)
#     println("In main method")
#     @show f
#     @show args
#     any(is_tracked, args) && push!(ctx.metadata, Op(f, map(untrack, args)...))
#     return track(f, args...; kwargs...)
# end

# function execute(ctx::FluxCtx, ::typeof(track), args...; kwargs...)
#     println("In fallback")
#     @show args
#     return OverdubInstead()
# end

# import Flux.Tracker: ∇broadcast
# function execute(ctx::FluxCtx, ::typeof(∇broadcast), f, args...)
#     println("In broadcast thingy")
#     any(is_tracked, args) && push!(ctx.metadata, Op(∇broadcast, f, map(untrack, args)...))
#     return OverdubInstead()
# end

# function execute(ctx::FluxCtx, ::typeof(track), f::Call, args...; kwargs...)
#     println("In call version")
#     @show f
#     @show typeof(f)
#     @show args
#     return OverdubInstead()
# end

# function execute(ctx::FluxCtx, ::typeof(track), ::typeof(sum), args...; kwargs...)
#     println("Inside shitty sum thing")
# end

# function execute(ctx::FluxCtx, ::typeof(track), ::typeof(sum), args...)
#     println("Inside shitty other sum thing.")
# end

# # Toy example
# baz = x->sin(cos(x))
# tape = Tape()
# overdub(FluxCtx(metadata=tape), Tracker.forward, baz, 5)

# tape = Tape()
# overdub(FluxCtx(metadata=tape), Tracker.forward, loss, 1.0)

tape = Tape();
overdub(FluxCtx(metadata=tape), Tracker.forward, x->5 * sum(x), randn(10));
tape

# tape = Tape();
# overdub(FluxCtx(metadata=tape), Tracker.forward, x->sum(abs.(x)), randn(10));




using Cassette

f(x;y) = x+y
Cassette.@context FooCtx
Cassette.execute(ctx::FooCtx, ::typeof(f), x; y)  = x+y+1
Cassette.execute(ctx::FooCtx, ::typeof(Core.kwfunc(f)), kw::Any, ::typeof(f), x) = 
    Core.kwfunc(Cassette.execute)(kw, Cassette.execute, ctx, f, x)

julia> Cassette.@overdub FooCtx() f(1;y=3)
5



