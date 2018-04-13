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
(c::CompositeMean)(X...) = c.f(map(x->x(X...), c.x)...)
mean(c::CompositeMean, X::AVM) = c.f.(map(x->mean(x, X), c.x)...)
cov(c::CompositeKernel, X::AVM) = c.f.(map(k->cov(k, X), c.x)...)
xcov(c::Union{CompositeKernel, CompositeCrossKernel}, X::AVM, X′::AVM) =
    c.f.(map(k->xcov(k, X, X′), c.x)...)

struct LhsCross <: CrossKernel
    f::MeanFunction
    k::CrossKernel
end
xcov(k::LhsCross, X::AVM, X′::AVM) = mean(k.f, X)' * xcov(k.k, X, X′)

struct RhsCross <: CrossKernel
    f::MeanFunction
    k::CrossKernel
end
xcov(k::RhsCross, X::AVM, X′::AVM) = xcov(k.k, X, X′) * mean(k.f, X′)

struct OuterCross <: CrossKernel
    f::MeanFunction
    f′::MeanFunction
    k::CrossKernel
end
xcov(k::OuterCross, X::AVM, X′::AVM) = mean(k.f, X)' * xcov(k.k, X, X′) * mean(k.f′, X′)

struct OuterKernel <: Kernel
    f::MeanFunction
    k::Kernel
end
cov(k::OuterKernel, X::AVM) = Xt_A_X(k.k, mean(k.f, X))
xcov(k::OuterKernel, X::AVM, X′::AVM) = mean(k.f, X)' * k.k * mean(k.f, Y)
