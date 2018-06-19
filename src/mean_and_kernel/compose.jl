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
map(f::CompositeMean, X::DataSet) = map(f.f, map(f->map(f, X), f.x)...)

# CompositeKernel definitions.
(k::CompositeKernel)(x, x′) = map(k.f, map(f->f(x, x′), k.x)...)
(k::CompositeKernel)(x) = map(k.f, map(f->f(x), k.x)...)
size(k::CompositeKernel, N::Int) = size(k.x[1], N)
isstationary(k::CompositeKernel) = all(map(isstationary, k.x))

map(f::CompositeKernel, X::DataSet) = map(f.f, map(f->map(f, X), f.x)...)
function pairwise(f::CompositeKernel, X::DataSet)
    return LazyPDMat(map(f.f, map(f->pairwise(f, X), f.x)...))
end
function map(f::CompositeKernel, X::DataSet, X′::DataSet)
    return map(f.f, map(f->map(f, X, X′), f.x)...)
end
function pairwise(f::CompositeKernel, X::DataSet, X′::DataSet)
    return map(f.f, map(f->pairwise(f, X, X′), f.x)...)
end

# CompositeCrossKernel definitions.
(k::CompositeCrossKernel)(x, x′) = map(k.f, map(f->f(x, x′), k.x)...)
size(k::CompositeCrossKernel, N::Int) = size(k.x[1], N)
isstationary(k::CompositeCrossKernel) = all(map(isstationary, k.x))

function map(f::CompositeCrossKernel, X::DataSet, X′::DataSet)
    return map(f.f, map(f->map(f, X, X′), f.x)...)
end
function pairwise(f::CompositeCrossKernel, X::DataSet, X′::DataSet)
    return map(f.f, map(f->pairwise(f, X, X′), f.x)...)
end



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
pairwise(k::LhsCross, X::DataSet, X′::DataSet) = map(k.f, X) .* pairwise(k.k, X, X′)

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
pairwise(k::RhsCross, X::DataSet, X′::DataSet) = pairwise(k.k, X, X′) .* map(k.f, X′)'

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
function pairwise(k::OuterCross, X::DataSet, X′::DataSet)
    return map(k.f, X) .* pairwise(k.k, X, X′) .* map(k.f, X′)'
end

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
function pairwise(k::OuterKernel, X::DataSet)
    return Xt_A_X(pairwise(k.k, X), Diagonal(map(k.f, X)))
end
function pairwise(k::OuterKernel, X::DataSet, X′::DataSet)
    return map(k.f, X) .* pairwise(k.k, X, X′) .* map(k.f, X′)'
end



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
+(k::Kernel, k′::Kernel) = CompositeKernel(+, k, k′)
+(k::Kernel, k′::Real) = +(promote(k, k′)...)
+(k::Real, k′::Kernel) = +(promote(k, k′)...)

*(k::Kernel, k′::Kernel) = CompositeKernel(*, k, k′)
*(k::Kernel, k′::Real) = *(promote(k, k′)...)
*(k::Real, k′::Kernel) = *(promote(k, k′)...)
