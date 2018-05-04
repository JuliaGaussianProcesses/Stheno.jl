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
length(c::CompositeMean) = length(c.x[1])
size(c::Union{CompositeKernel, CompositeCrossKernel}, N::Int) = size(c.x[1], N)
mean(c::CompositeMean, X::AVM) = map(c.f, map(μ->mean(μ, X), c.x)...)
cov(c::CompositeKernel, X::AVM) = LazyPDMat(c.f.(map(k->xcov(k, X), c.x)...))
cov(c::CompositeKernel{typeof(+)}, X::AVM) = LazyPDMat(sum(map(k->xcov(k, X), c.x)))
xcov(c::Union{CompositeKernel, CompositeCrossKernel}, X::AVM, X′::AVM) =
    map(c.f, map(k->xcov(k, X, X′), c.x)...)
marginal_cov(c::CompositeKernel, X::AVM) = map(c.f, map(k->marginal_cov(k, X), c.x)...)

"""
    LhsCross <: CrossKernel

A cross-kernel given by `k(x, x′) = f(x) * k(x, x′)`.
"""
struct LhsCross <: CrossKernel
    f::MeanFunction
    k::CrossKernel
end
size(k::LhsCross, N::Int) = size(k.k, N)
xcov(k::LhsCross, X::AVM, X′::AVM) = Diagonal(mean(k.f, X)) * xcov(k.k, X, X′)

"""
    RhsCross <: CrossKernel

A cross-kernel given by `k(x, x′) = k(x, x′) * f(x′)`.
"""
struct RhsCross <: CrossKernel
    k::CrossKernel
    f::MeanFunction
end
size(k::RhsCross, N::Int) = size(k.k, N)
xcov(k::RhsCross, X::AVM, X′::AVM) = xcov(k.k, X, X′) * Diagonal(mean(k.f, X′))

"""
    OuterCross <: CrossKernel

A cross-kernel given by `k(x, x′) = f(x) * k(x, x′) * f′(x′)`.
"""
struct OuterCross <: CrossKernel
    f::MeanFunction
    k::CrossKernel
    f′::MeanFunction
end
size(k::OuterCross, N::Int) = size(k.k, N)
xcov(k::OuterCross, X::AVM, X′::AVM) =
    Diagonal(mean(k.f, X)) * xcov(k.k, X, X′) * Diagonal(mean(k.f′, X′))

"""
    OuterKernel <: Kernel

A kernel given by `k(x, x′) = f(x) * k(x, x′) * f(x′)`.
"""
struct OuterKernel <: Kernel
    f::MeanFunction
    k::Kernel
end
size(k::OuterKernel, N::Int) = size(k.k, N)
cov(k::OuterKernel, X::AVM) = Xt_A_X(cov(k.k, X), Diagonal(mean(k.f, X)))
xcov(k::OuterKernel, X::AVM, X′::AVM) =
    Diagonal(mean(k.f, X)) * xcov(k.k, X, X′) * Diagonal(mean(k.f, X′))
marginal_cov(k::OuterKernel, X::AVM) = marginal_cov(k.k, X) .* mean(k.f, X).^2


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
