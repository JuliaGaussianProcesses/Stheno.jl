# Create CompositeMean, CompositeKernel and CompositeCrossKernel.
for (composite_type, parent_type) in [(:CompositeMean, :MeanFunction),
                                      (:CompositeKernel, :Kernel),
                                      (:CompositeCrossKernel, :CrossKernel),]
@eval struct $composite_type{Tf, N} <: $parent_type
    f::Tf
    x::Tuple{Vararg{Any, N}}
    $composite_type(f::Tf, x::Vararg{Any, N}) where {Tf, N} = new{Tf, N}(f, x)
end
end

# CompositeMean definitions.
eachindex(c::CompositeMean) = eachindex(c.x[1])
length(c::CompositeMean) = length(c.x[1])
(μ::CompositeMean)(x) = map(μ.f, map(f->f(x), μ.x)...)
_map(f::CompositeMean, X::AV) = map(f.f, map(f->map(f, X), f.x)...)
map(f::CompositeMean, ::Colon) = map(f.f, map(f->map(f, :), f.x)...)

# CompositeKernel definitions.
(k::CompositeKernel)(x, x′) = map(k.f, map(f->f(x, x′), k.x)...)
(k::CompositeKernel)(x) = map(k.f, map(f->f(x), k.x)...)
size(k::CompositeKernel, N::Int) = size(k.x[1], N)
isstationary(k::CompositeKernel) = all(map(isstationary, k.x))
@noinline eachindex(k::CompositeKernel) = eachindex(k.x[1], 1)

_map(f::CompositeKernel, X::AV) = map(f.f, map(f->map(f, X), f.x)...)
_pairwise(f::CompositeKernel, X::AV) = LazyPDMat(map(f.f, map(f->pairwise(f, X), f.x)...))
_map(f::CompositeKernel, X::AV, X′::AV) = map(f.f, map(f->map(f, X, X′), f.x)...)
_pairwise(f::CompositeKernel, X::AV, X′::AV) = map(f.f, map(f->pairwise(f, X, X′), f.x)...)

map(f::CompositeKernel, ::Colon) = map(f.f, map(f->map(f, :), f.x)...)
pairwise(f::ConstantKernel, ::Colon) = LazyPDMat(map(f.f, map(f->pairwise(f, :), f.x)...))

# CompositeCrossKernel definitions.
(k::CompositeCrossKernel)(x, x′) = map(k.f, map(f->f(x, x′), k.x)...)
size(k::CompositeCrossKernel, N::Int) = size(k.x[1], N)
isstationary(k::CompositeCrossKernel) = all(map(isstationary, k.x))
eachindex(k::CompositeCrossKernel, dim::Int) = eachindex(k.x[1], dim)

_map(f::CompositeCrossKernel, X::AV, X′::AV) = map(f.f, map(f->map(f, X, X′), f.x)...)
function _pairwise(f::CompositeCrossKernel, X::AV, X′::AV)
    return map(f.f, map(f->pairwise(f, X, X′), f.x)...)
end

map(f::CompositeCrossKernel, ::Colon) = map(f.f, map(f->map(f, :), f.x)...)
pairwise(f::CompositeCrossKernel, ::Colon) = map(f.f, map(f->pairwise(f, :), f.x)...)


############################## Multiply-by-Function Kernels ################################

"""
    LhsCross <: CrossKernel

A cross-kernel given by `k(x, x′) = f(x) * k(x, x′)`.
"""
struct LhsCross <: CrossKernel
    f::Any
    k::CrossKernel
end
(k::LhsCross)(x, x′) = k.f(x) * k.k(x, x′)
size(k::LhsCross, N::Int) = size(k.k, N)
_pairwise(k::LhsCross, X::AV, X′::AV) = map(k.f, X) .* pairwise(k.k, X, X′)
eachindex(k::LhsCross, dim::Int) = eachindex(k.k, dim)

# map(k::LhsCross, ::Colon) = map(k.f, :) .* map(k.k, :)
# pairwise(k::LhsCross, ::Colon) = map(k.f, :) .* pairwise(k.k, :)

"""
    RhsCross <: CrossKernel

A cross-kernel given by `k(x, x′) = k(x, x′) * f(x′)`.
"""
struct RhsCross <: CrossKernel
    k::CrossKernel
    f
end
(k::RhsCross)(x, x′) = k.k(x, x′) * k.f(x′)
size(k::RhsCross, N::Int) = size(k.k, N)
_pairwise(k::RhsCross, X::AV, X′::AV) = pairwise(k.k, X, X′) .* map(k.f, X′)'
eachindex(k::RhsCross, dim::Int) = eachindex(k.k, dim)

# map(k::RhsCross, ::Colon) = map(k.k, :) .* map(k.f, :)
# pairwise(k::RhsCross, ::Colon) = pairwise(k.k, :) .* map(k.f, :)'

"""
    OuterCross <: CrossKernel

A kernel given by `k(x, x′) = f(x) * k(x, x′) * f(x′)`.
"""
struct OuterCross <: CrossKernel
    f
    k::CrossKernel
end
(k::OuterCross)(x, x′) = k.f(x) * k.k(x, x′) * k.f(x′)
size(k::OuterCross, N::Int) = size(k.k, N)
function _pairwise(k::OuterCross, X::AV, X′::AV)
    return map(k.f, X) .* pairwise(k.k, X, X′) .* map(k.f, X′)'
end
eachindex(k::OuterCross, dim::Int) = eachindex(k.k, dim)

# map(k::OuterCross, ::Colon) = map(k.f, :) .* pairwise(k.k, :) .* map(k.f, :)'

"""
    OuterKernel <: Kernel

A kernel given by `k(x, x′) = f(x) * k(x, x′) * f(x′)`.
"""
struct OuterKernel <: Kernel
    f
    k::Kernel
end
(k::OuterKernel)(x, x′) = k.f(x) * k.k(x, x′) * k.f(x′)
(k::OuterKernel)(x) = k.f(x)^2 * k.k(x)
size(k::OuterKernel, N::Int) = size(k.k, N)
_pairwise(k::OuterKernel, X::AV) = Xt_A_X(pairwise(k.k, X), Diagonal(map(k.f, X)))
function _pairwise(k::OuterKernel, X::AV, X′::AV)
    return map(k.f, X) .* pairwise(k.k, X, X′) .* map(k.f, X′)'
end
eachindex(k::OuterKernel) = eachindex(k.k)


############################## Convenience functionality ##############################

import Base: +, *, promote_rule, convert

promote_rule(::Type{<:MeanFunction}, ::Type{<:Union{Real, Function}}) = MeanFunction
convert(::Type{MeanFunction}, x::Real) = ConstantMean(x)

promote_rule(::Type{<:Kernel}, ::Type{<:Union{Real, Function}}) = Kernel
convert(::Type{<:CrossKernel}, x::Real) = ConstantKernel(x)

# Composing mean functions.
+(μ::MeanFunction, μ′::MeanFunction) = CompositeMean(+, μ, μ′)
+(μ::MeanFunction, μ′::Real) = +(promote(μ, μ′)...)
+(μ::Real, μ′::MeanFunction) = +(promote(μ, μ′)...)

*(μ::MeanFunction, μ′::MeanFunction) = CompositeMean(*, μ, μ′)
*(μ::MeanFunction, μ′::Real) = *(promote(μ, μ′)...)
*(μ::Real, μ′::MeanFunction) = *(promote(μ, μ′)...)

# Composing kernels.
+(k::CrossKernel, k′::CrossKernel) = CompositeKernel(+, k, k′)
+(k::CrossKernel, k′::Real) = +(promote(k, k′)...)
+(k::Real, k′::CrossKernel) = +(promote(k, k′)...)

*(k::CrossKernel, k′::CrossKernel) = CompositeKernel(*, k, k′)
*(k::CrossKernel, k′::Real) = *(promote(k, k′)...)
*(k::Real, k′::CrossKernel) = *(promote(k, k′)...)
