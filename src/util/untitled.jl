using Revise
using LinearAlgebra
using LinearAlgebra: RealHermSymComplexHerm, copytri!
import LinearAlgebra: cholesky, UpperTriangular, logdet, \, /
const chol_type = Union{StridedMatrix, RealHermSymComplexHerm{<:Real, <:StridedMatrix}}

using Flux.Tracker
using Flux.Tracker: tracker, Call, track, TrackedVecOrMat, Tracked, TrackedArray,
    TrackedReal

# # This probably isn't optimal, but probably isn't horrific.
# function chol_sensitivity(
#     U::AbstractMatrix{T},
#     Ū::AbstractMatrix{T},
#     Σ::AbstractMatrix{T},
# ) where T<:Real
#     Σ̄ = Ū * UpperTriangular(U)'
#     Σ̄ = copytri!(Σ̄, 'U')
#     Σ̄ = ldiv!(UpperTriangular(U), Σ̄)
#     BLAS.trsm!('R', 'U', 'T', 'N', one(T), U, Σ̄)
#     @inbounds for n in diagind(Σ̄)
#         Σ̄[n] *= 0.5
#     end
#     return UpperTriangular(Σ̄)
# end

# # This is a fairly incomplete implementation of the Cholesky that just about works in the
# # most basic situation. It should, in principle, be straightforward to generalise.
# function cholesky(Σ::TrackedMatrix{<:Real, <:chol_type})

#     # Compute the Cholesky factorisation as per usual.
#     C_raw = cholesky(Tracker.data(Σ))
#     U = C_raw.factors

#     # Compute reverse-pass function.
#     C̄ = Ū->(chol_sensitivity(Tracker.data(U), Tracker.data(Ū), Tracker.data(Σ)),)

#     # Track factors from Cholesky factorisation. Nothing else needs to be touched.
#     tracked_factors = track(Call(C̄, (tracker(Σ),)), U)

#     # Pack everything into a Cholesky.
#     return Cholesky{eltype(Σ), typeof(tracked_factors)}(tracked_factors, C_raw.uplo, C_raw.info)
# end

# # This is also a hack, but does the job nicely.
# logdet(C::Cholesky{<:Real, <:TrackedMatrix}) = 2 * sum(log, C.factors[diagind(C.factors)])

# # Sensible implementation of UpperTriangular.
# using Flux.Tracker: @grad

# # U = UpperTriangular(X)
# UpperTriangular(X::TrackedMatrix) = Tracker.track(UpperTriangular, X)
# @grad function UpperTriangular(X)
#     return UpperTriangular(Tracker.data(X)), function (Ū)
#         return (UpperTriangular(copy(Ū)),)
#     end
# end

# # L = LowerTriangular(X)
# LowerTriangular(X::TrackedMatrix) = Tracker.track(LowerTriangular, X)
# @grad function LowerTriangular(X)
#     return LowerTriangular(Tracker.data(X)), function(L̄)
#         return (LowerTriangular(copy(L̄)),)
#     end
# end

# # Y = A \ B
# \(A::TrackedMatrix, B::TrackedVecOrMat) = Tracker.track(\, A, B)
# \(A::AbstractMatrix, B::TrackedVecOrMat) = Tracker.track(\, A, B)
# \(A::TrackedMatrix, B::AbstractVecOrMat) = Tracker.track(\, A, B)
# @grad function \(A::AbstractMatrix, B::AbstractVector)
#     Y = Tracker.data(A) \ Tracker.data(B)
#     return Y, function(Ȳ)
#         B̄ = A' \ Ȳ
#         return (-B̄ * Y', B̄)
#     end
# end

# # Y = A / B
# /(A::TrackedVecOrMat, B::TrackedMatrix) = Tracker.track(/, A, B)
# /(A::AbstractVecOrMat, B::TrackedMatrix) = Tracker.track(/, A, B)
# /(A::TrackedVecOrMat, B::AbstractMatrix) = Tracker.track(/, A, B)
# @grad function /(A::AbstractVecOrMat, B::AbstractMatrix)
#     Y = Tracker.data(A) / Tracker.data(B)
#     return Y, function(Ȳ)
#         Ā = Ȳ / B'
#         return (Ā, -Y' * Ā)
#     end
# end


# N = 4
# B = randn(N, N)
# x = randn(N)
# A = B' * B
# C = cholesky(Symmetric(A))

# foo(A) = logdet(cholesky(A))
# bar(A) = sum(abs2, cholesky(A).U \ x)


# using Flux.Tracker: forward
# l, Ā = forward(foo, A)
# Ā(1.0)[1]

# l, Ā = forward(bar, A)
# Ā(1.0)[1]

# using Test




# using ForwardDiff: gradient
# gradient(foo, A)
# gradient(bar, A)



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
    Op{Tf, Tvalue, Targs, Tkwargs}

The totality of a call to a (pure) primtive function `f` at `args` and `kwargs`,
producing `value`.
"""
struct Op{Tf, Tvalue, Targs, Tkwargs}
    f::Tf
    value::Tvalue
    args::Targs
    kwargs::Tkwargs
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

const Tape = Vector{Op}

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
using Core: kwftype
using Cassette: @context, overdub, OverdubInstead, enabletagging
import Cassette: prehook, execute, posthook
@context FluxCtx

# Define what is tracked and what isn't.
is_tracked(::Any) = false
is_tracked(::TrackedArray) = true
is_tracked(::TrackedReal) = true

untrack(x::Union{TrackedReal, TrackedArray}) = x.data
untrack(x) = x

function posthook(ctx::FluxCtx, tmp, ::kwftype(typeof(track)), kwargs, ::typeof(track), f, args...)
    push!(ctx.metadata, Op(f, tmp, args, kwargs))
end
function posthook(ctx::FluxCtx, tmp, ::typeof(track), f, args...)
    push!(ctx.metadata, Op(f, tmp, args, nothing))
end
posthook(ctx::FluxCtx, tmp, ::kwftype(typeof(track)), kwargs, ::typeof(track), ::Call, args...) = nothing
posthook(ctx::FluxCtx, tmp, ::typeof(track), ::Call, args...) = nothing

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

using Stheno: pairwise
tape = Tape();
P, Q, D = 100, 200, 10;
X, Y = randn(D, P), randn(D, Q);
overdub(
    FluxCtx(metadata=tape),
    Tracker.forward,
    (X, Y)->pairwise(EQ(), ColsAreObs(X), ColsAreObs(Y)),
    X, Y,
)
tape

using BenchmarkTools, Distances

# D, back = Tracker._forward(pairwise, SqEuclidean(), X, Y);

# @benchmark pairwise(SqEuclidean(), $X, $Y)
# @benchmark Tracker._forward(pairwise, SqEuclidean(), $X, $Y)

# D̄ = fill(1.0, size(D));
# @benchmark back($D̄)


# D, back = Tracker._forward(pairwise, SqEuclidean(), X);
# D̄ = fill(1.0, size(D));

# @benchmark pairwise(SqEuclidean(), $X)
# @benchmark Tracker._forward(pairwise, SqEuclidean(), $X)

# @benchmark back($D̄)


x, y = randn(P), randn(Q)
D, back = Tracker._forward(pairwise, SqEuclidean(), x, y);
D̄ = fill(1.0, size(D));

X, Y = reshape(x, 1, length(x)), reshape(y, 1, length(y));


@benchmark pairwise(SqEuclidean(), $x, $y)
@benchmark pairwise(SqEuclidean(), $X, $Y)

@benchmark Tracker._forward(pairwise, SqEuclidean(), $x, $y)
@benchmark Tracker._forward(pairwise, SqEuclidean(), $X, $Y)

@benchmark back($D̄)
out_xy = back(D̄)

D, back = Tracker._forward(pairwise, SqEuclidean(), X, Y);
@benchmark back($D̄)

out_XY = back(D̄)

# f(x; y) = x+y
# Cassette.@context FooCtx
# Cassette.execute(ctx::FooCtx, ::typeof(f), x; y)  = x+y+1
# Cassette.execute(ctx::FooCtx, ::typeof(Core.kwfunc(f)), kw::Any, ::typeof(f), x) = 
#     Core.kwfunc(Cassette.execute)(kw, Cassette.execute, ctx, f, x)

# julia> Cassette.@overdub FooCtx() f(1;y=3)
# 5

# using Core: kwftype
# function execute(ctx::FluxCtx, ::kwftype(typeof(f)), kwargs::Any, ::typeof(f), args...)
#     return Cassette.execute(ctx, f, args...; kwargs...)
# end

# Naive implementation (I think).
julia> @benchmark back($D̄)
BenchmarkTools.Trial: 
  memory estimate:  121.53 KiB
  allocs estimate:  24
  --------------
  minimum time:     72.786 μs (0.00% GC)
  median time:      75.986 μs (0.00% GC)
  mean time:        85.634 μs (10.59% GC)
  maximum time:     2.023 ms (91.87% GC)
  --------------
  samples:          10000
  evals/sample:     1

# Slightly less naive.
julia> @benchmark back($D̄)
BenchmarkTools.Trial: 
  memory estimate:  105.78 KiB
  allocs estimate:  23
  --------------
  minimum time:     71.024 μs (0.00% GC)
  median time:      74.116 μs (0.00% GC)
  mean time:        82.504 μs (9.38% GC)
  maximum time:     2.030 ms (91.54% GC)
  --------------
  samples:          10000
  evals/sample:     1

# Marginally less naive again.
julia> @benchmark back($D̄)
BenchmarkTools.Trial: 
  memory estimate:  97.84 KiB
  allocs estimate:  22
  --------------
  minimum time:     70.347 μs (0.00% GC)
  median time:      73.588 μs (0.00% GC)
  mean time:        81.789 μs (9.31% GC)
  maximum time:     2.037 ms (91.16% GC)
  --------------
  samples:          10000
  evals/sample:     1

# Fuse some stuff.
julia> @benchmark back($D̄)
BenchmarkTools.Trial: 
memory estimate:  74.16 KiB
allocs estimate:  20
--------------
minimum time:     67.989 μs (0.00% GC)
median time:      69.936 μs (0.00% GC)
mean time:        77.438 μs (8.47% GC)
maximum time:     2.218 ms (94.07% GC)
--------------
samples:          10000
evals/sample:     1



############# Load some stuff ################

using Revise
using Stheno, Flux, Flux.Tracker, Random, BenchmarkTools, Distances, Zygote
using Stheno: @model
using Distances: SqEuclidean

Ns = [10, 100, 1_000, 10_000]
Ns = [10, 100, 1_000]


############# Check gradient of pairwise(SqEuclidean()...) ##########

rng = MersenneTwister(123456);
X = randn(rng, 2, 10);
obj(X) = sum(Stheno.pairwise(SqEuclidean(), X))

println("Forward")
@benchmark obj($X)

println("Flux")
@benchmark Flux.gradient(obj, $X)

println("Zygote")
@benchmark Zygote.gradient(obj, $X)



############ Check gradient of broadcasted operations over a distance matrix ###############

rng = MersenneTwister(123456);
D = pairwise(SqEuclidean(), randn(rng, 2, 100));
obj_pw(D) = sum(-0.5 .* D)

for N in Ns
    println(N)
    D = pairwise(SqEuclidean(), randn(rng, 2, N));

    println("Forward")
    display(@benchmark obj_pw($D))

    println("Flux")
    display(@benchmark Flux.gradient(obj_pw, $D))

    println("Zygote")
    display(@benchmark Zygote.gradient(obj_pw, $D))
end



############# Check gradient of EQ covariance #############

println("EQ covariance")
rng = MersenneTwister(123456);
# obj_eq(x) = sum(Stheno._pw(EQ(), ColsAreObs(x)))
obj_eq(x) = sum(Stheno._pw(EQ(), x))

for N in Ns
    println(N)
    X = randn(rng, 2, N)
    x = randn(rng, N)

    println("Forward")
    display(@benchmark obj_eq($x))

    println("Flux")
    display(@benchmark Flux.gradient(obj_eq, $x))

    println("Zygote")
    display(@benchmark Zygote.gradient(obj_eq, $x))
end



############# Check gradient of logpdf w.r.t. inputs of a very simple model ##############

println("Gradient w.r.t. inputs")
rng = MersenneTwister(123456);
obj_simple(X, y) = logpdf(GP(EQ(), GPC())(ColsAreObs(X)), y)

for N in Ns
    println(N)
    X = randn(rng, 2, N)
    y = rand(rng, GP(EQ(), GPC())(ColsAreObs(X)))

    println("Forward")
    display(@benchmark (X->obj_simple(X, $y))($X))

    println("Flux")
    display(@benchmark Flux.gradient(X->obj_simple(X, $y), $X))
    display(@benchmark Flux.gradient(y->obj_simple($X, y), $y))
    display(@benchmark Flux.gradient(obj_simple, $X, $y))

    # println("Zygote")
    # display(@benchmark Zygote.gradient(X->obj_simple(X, $y), $X))
    # display(@benchmark Zygote.gradient(y->obj_simple($X, y), $y))
    # display(@benchmark Zygote.gradient(obj_simple, $X, $y))
end


############# Check gradient of model ###############

@model function foo(log_σ::Real)
    return exp(log_σ) * GP(EQ())
end

rng = MersenneTwister(123456);

function obj_logpdf(log_σ::Real, x, y)
    return logpdf(foo(log_σ)(ColsAreObs(x)), y)
end

for N in Ns
    x = randn(rng, 2, N)
    y = rand(rng, foo(0.0)(ColsAreObs(x)))
    println(N)

    println("Forward")
    display(@benchmark obj_logpdf(0.0, $x, $y))

    println("Flux")
    display(@benchmark Flux.gradient(logσ->obj_logpdf(logσ, $x, $y), 0.0))

    # println("Zygote")
    # display(@benchmark Zygote.gradient((x, y)->obj_logpdf(0.0, x, y), $x, $y))
end
