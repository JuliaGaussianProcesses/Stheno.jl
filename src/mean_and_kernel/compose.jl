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
mean(c::CompositeMean, X::AVM) = c.f.(map(μ->mean(μ, X), c.x)...)
cov(c::CompositeKernel, X::AVM) = c.f.(map(k->cov(k, X), c.x)...)
xcov(c::Union{CompositeKernel, CompositeCrossKernel}, X::AVM, X′::AVM) =
    map(c.f, map(k->xcov(k, X, X′), c.x)...)

"""
    LhsCross <: CrossKernel

A cross-kernel given by `k(x, x′) = f(x) * k(x, x′)`.
"""
struct LhsCross <: CrossKernel
    f::MeanFunction
    k::CrossKernel
end
xcov(k::LhsCross, X::AVM, X′::AVM) = Diagonal(mean(k.f, X)) * xcov(k.k, X, X′)

"""
    RhsCross <: CrossKernel

A cross-kernel given by `k(x, x′) = k(x, x′) * f(x′)`.
"""
struct RhsCross <: CrossKernel
    k::CrossKernel
    f::MeanFunction
end
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
cov(k::OuterKernel, X::AVM) = Xt_A_X(cov(k.k, X), Diagonal(mean(k.f, X)))
xcov(k::OuterKernel, X::AVM, X′::AVM) =
    Diagonal(mean(k.f, X)) * xcov(k.k, X, X′) * Diagonal(mean(k.f, X′))
