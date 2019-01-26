"""
    UnaryMean{Top, Tμ} <: MeanFunction

The mean function given by `op(μ(x))`.
"""
struct UnaryMean{Top, Tμ} <: MeanFunction
    op::Top
    μ::Tμ
end
_map(μ::UnaryMean, x::AV) = bcd(μ.op, _map(μ.μ, x))


"""
    BinaryMeanOp{Top, Tμ₁, Tμ₂} <: MeanFunction

The mean function given by `op(μ₁(x), μ₂(x))`.
"""
struct BinaryMean{Top, Tμ₁, Tμ₂} <: MeanFunction
    op::Top
    μ₁::Tμ₁
    μ₂::Tμ₂
end
_map(μ::BinaryMean, x::AV) = bcd(μ.op, _map(μ.μ₁, x), _map(μ.μ₂, x))


"""
    BinaryKernel{Top, Tk₁, Tk₂} <: Kernel

The kernel given by `op(k₁(x, x′), k₂(x, x′))`.
"""
struct BinaryKernel{Top, Tk₁<:Kernel, Tk₂<:Kernel} <: Kernel
    op::Top
    k₁::Tk₁
    k₂::Tk₂
end

# Binary operations.
_map(k::BinaryKernel, x::AV, x′::AV) = bcd(k.op, _map(k.k₁, x, x′), _map(k.k₂, x, x′))
_pw(k::BinaryKernel, x::AV, x′::AV) = bcd(k.op, _pw(k.k₁, x, x′), _pw(k.k₂, x, x′))

# Unary operations.
_map(k::BinaryKernel, x::AV) = bcd(k.op, _map(k.k₁, x), _map(k.k₂, x))
_pw(k::BinaryKernel, x::AV) = bcd(k.op, _pw(k.k₁, x), _pw(k.k₂, x))


"""
    BinaryCrossKernel{Top, Tk₁, Tk₂} <: CrossKernel

The cross kernel given by `op(k₁(x, x′), k₂(x, x′))`.
"""
struct BinaryCrossKernel{Top, Tk₁<:CrossKernel, Tk₂<:CrossKernel} <: CrossKernel
    op::Top
    k₁::Tk₁
    k₂::Tk₂
end

# Binary operations.
_map(k::BinaryCrossKernel, x::AV, x′::AV) = bcd(k.op, _map(k.k₁, x, x′), _map(k.k₂, x, x′))
_pw(k::BinaryCrossKernel, x::AV, x′::AV) = bcd(k.op, _pw(k.k₁, x, x′), _pw(k.k₂, x, x′))

# Unary operations.
_map(k::BinaryCrossKernel, x::AV) = bcd(k.op, _map(k.k₁, x), _map(k.k₂, x))
_pw(k::BinaryCrossKernel, x::AV) = bcd(k.op, _pw(k.k₁, x), _pw(k.k₂, x))



############################## Multiply-by-Function Kernels ################################


"""
    LhsCross <: CrossKernel{Tf, Tk<:CrossKernel}

A cross-kernel given by `f(x) * k(x, x′)`.
"""
struct LhsCross{Tf<:MeanFunction, Tk<:CrossKernel} <: CrossKernel
    f::Tf
    k::Tk
end
_map(k::LhsCross, x::AV, x′::AV) = bcd(*, map(k.f, x), _map(k.k, x, x′))
_pw(k::LhsCross, x::AV, x′::AV) = bcd(*, map(k.f, x), _pw(k.k, x, x′))

    
"""
    RhsCross <: CrossKernel{Tk<:CrossKernel, Tf}

A cross-kernel given by `k(x, x′) * f(x′)`.
"""
struct RhsCross{Tk<:CrossKernel, Tf<:MeanFunction} <: CrossKernel
    k::Tk
    f::Tf
end
_map(k::RhsCross, x::AV, x′::AV) = bcd(*, _map(k.k, x, x′), map(k.f, x′))
_pw(k::RhsCross, x::AV, x′::AV) = bcd(*, _pw(k.k, x, x′), map(k.f, x′)')


"""
    OuterCross <: CrossKernel{Tf, Tk<:CrossKernel}

A kernel given by `f(x) * k(x, x′) * f(x′)`.
"""
struct OuterCross{Tf<:MeanFunction, Tk<:CrossKernel} <: CrossKernel
    f::Tf
    k::Tk
end
_map(k::OuterCross, x::AV, x′::AV) = bcd(*, map(k.f, x), _map(k.k, x, x′), map(k.f, x′))
_pw(k::OuterCross, x::AV, x′::AV) = bcd(*, map(k.f, x), _pw(k.k, x, x′), map(k.f, x′)')


"""
    OuterKernel <: Kernel

A kernel given by `f(x) * k(x, x′) * f(x′)`.
"""
struct OuterKernel{Tf<:MeanFunction, Tk<:Kernel} <: Kernel
    f::Tf
    k::Tk
end

# Binary methods.
function _map(k::OuterKernel, x::AV, x′::AV)
    return bcd(*, map(k.f, x), _map(k.k, x, x′), map(k.f, x′))
end
function _pw(k::OuterKernel, x::AV, x′::AV)
    return bcd(*, map(k.f, x), _pw(k.k, x, x′), map(k.f, x′)')
end

# Unary methods.
function _map(k::OuterKernel, x::AV)
    fx = map(k.f, x)
    return bcd(*, fx, _map(k.k, x), fx)
end
function _pw(k::OuterKernel, x::AV)
    fx = map(k.f, x)
    return bcd(*, fx, _pw(k.k, x), fx')
end
