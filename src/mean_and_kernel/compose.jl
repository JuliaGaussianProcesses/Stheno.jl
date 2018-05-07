# # Create CompositeMean, CompositeKernel and CompositeCrossKernel.
# for (composite_type, parent_type) in [(:CompositeMean, :MeanFunction),
#                                       (:CompositeKernel, :Kernel),
#                                       (:CompositeCrossKernel, :CrossKernel),]
# @eval struct $composite_type{Tf, N} <: $parent_type
#     f::Tf
#     x::Tuple{Vararg{Any, N}}
#     $composite_type(f::Tf, x::Vararg{Any, N}) where {Tf, N} = new{Tf, N}(f, x)
# end
# end
# length(c::CompositeMean) = length(c.x[1])
# size(c::Union{CompositeKernel, CompositeCrossKernel}, N::Int) = size(c.x[1], N)
# mean(c::CompositeMean, X::AVM) = map(c.f, map(μ->mean(μ, X), c.x)...)
# cov(c::CompositeKernel, X::AVM) = LazyPDMat(c.f.(map(k->xcov(k, X), c.x)...))
# cov(c::CompositeKernel{typeof(+)}, X::AVM) = LazyPDMat(sum(map(k->xcov(k, X), c.x)))
# xcov(c::Union{CompositeKernel, CompositeCrossKernel}, X::AVM, X′::AVM) =
#     map(c.f, map(k->xcov(k, X, X′), c.x)...)
# marginal_cov(c::CompositeKernel, X::AVM) = map(c.f, map(k->marginal_cov(k, X), c.x)...)

"""
    MapReduce{Tf, N} <: MeanFunctionOrKernel

Maps the functions `fs` over the arguments, and reduces using `op`. Note that this is the
opposite way around to the usual MapReduce.
"""
struct MapReduce{N, Top, A} <: MeanFunctionOrKernel
    op::Top
    fs::Tuple{Vararg{Any, A}}
    MapReduce(N::Int, op::Top, fs::Vararg{Any, A}) where {Top, A} = new{N, Top, A}(op, fs)
end
length(c::MapReduce) = length(c.fs[1])
size(c::MapReduce, n::Int) = size(c.fs[1], n)
(f::MapReduce{N})(x::Vararg{Any, N}) where N = mapreduce(f->f(x...), f.op, f.fs)
unary_colwise(f::MapReduce{1}, X::AMRV) = map(f.op, map(f->unary_colwise(f, X), f.fs)...)
binary_colwise(f::MapReduce{2}, X::AMRV, Y::AMRV) =
    map(f.op, map(f->binary_colwise(f, X, Y), f.fs)...)
pairwise(f::MapReduce{2}, X::AMRV) = map(f.op, map(f->pairwise(f, X), f.fs)...)
pairwise(f::MapReduce{2}, X::AMRV, Y::AMRV) = map(f.op, map(f->pairwise(f, X, Y), f.fs)...)

"""
    LhsCross <: Kernel

A cross-kernel given by `k(x, x′) = f(x) * k(x, x′)`.
"""
struct LhsCross <: Kernel
    f::Any
    k::Any
end
(k::LhsCross)(x, x′) = k.f(x) * k.k(x, x′)
size(k::LhsCross, N::Int) = size(k.k, N)
pairwise(k::LhsCross, X::AMRV, X′::AMRV) = unary_colwise(k.f, X) .* pairwise(k.k, X, X′)

"""
    RhsCross <: Kernel

A cross-kernel given by `k(x, x′) = k(x, x′) * f(x′)`.
"""
struct RhsCross <: Kernel
    k
    f
end
(k::RhsCross)(x, x′) = k.k(x, x′) * k.f(x′)
size(k::RhsCross, N::Int) = size(k.k, N)
pairwise(k::RhsCross, X::AMRV, X′::AMRV) = pairwise(k.k, X, X′) .* unary_colwise(k.f, X′)'

"""
    Outer <: Kernel

A kernel given by `k(x, x′) = f(x) * k(x, x′) * f(x′)`.
"""
struct Outer <: Kernel
    f
    k
end
(k::Outer)(x, x′) = k.f(x) * k.k(x, x′) * k.f(x′)
size(k::Outer, N::Int) = size(k.k, N)
pairwise(k::Outer, X::AMRV) = Xt_A_X(pairwise(k.k, X), Diagonal(unary_colwise(k.f, X)))
pairwise(k::Outer, X::AMRV, X′::AMRV) =
    unary_colwise(k.f, X) .* pairwise(k.k, X, X′) .* unary_colwise(k.f, X′)'


############################## Convenience functionality ##############################

import Base: +, *, promote_rule, convert

promote_rule(::Type{<:MeanFunction}, ::Type{<:Union{Real, Function}}) = MeanFunction
convert(::Type{MeanFunction}, x::Real) = Const(1, x)

promote_rule(::Type{<:Kernel}, ::Type{<:Union{Real, Function}}) = Kernel
convert(::Type{<:Kernel}, x::Real) = Const(2, x)

# Composing mean functions.
+(μ::MeanFunction, μ′::MeanFunction) = MapReduce(1, +, μ, μ′)
+(μ::MeanFunction, μ′::Real) = +(promote(μ, μ′)...)
+(μ::Real, μ′::MeanFunction) = +(promote(μ, μ′)...)

*(μ::MeanFunction, μ′::MeanFunction) = MapReduce(1, *, μ, μ′)
*(μ::MeanFunction, μ′::Real) = *(promote(μ, μ′)...)
*(μ::Real, μ′::MeanFunction) = *(promote(μ, μ′)...)

# Composing kernels.
+(k::Kernel, k′::Kernel) = MapReduce(2, +, k, k′)
+(k::Kernel, k′::Real) = +(promote(k, k′)...)
+(k::Real, k′::Kernel) = +(promote(k, k′)...)

*(k::Kernel, k′::Kernel) = MapReduce(2, *, k, k′)
*(k::Kernel, k′::Real) = *(promote(k, k′)...)
*(k::Real, k′::Kernel) = *(promote(k, k′)...)
