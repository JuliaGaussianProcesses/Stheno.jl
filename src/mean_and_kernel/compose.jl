"""
    UnaryMean{Top, Tμ} <: MeanFunction

The mean function given by `op(μ(x))`.
"""
struct UnaryMean{Top, Tμ} <: MeanFunction
    op::Top
    μ::Tμ
end
ew(μ::UnaryMean, x::AV) = μ.op.(ew(μ.μ, x))


"""
    BinaryMeanOp{Top, Tμ₁, Tμ₂} <: MeanFunction

The mean function given by `op(μ₁(x), μ₂(x))`.
"""
struct BinaryMean{Top, Tμ₁, Tμ₂} <: MeanFunction
    op::Top
    μ₁::Tμ₁
    μ₂::Tμ₂
end
ew(μ::BinaryMean, x::AV) = μ.op.(ew(μ.μ₁, x), ew(μ.μ₂, x))


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
ew(k::BinaryKernel, x::AV, x′::AV) = k.op.(ew(k.k₁, x, x′), ew(k.k₂, x, x′))
pw(k::BinaryKernel, x::AV, x′::AV) = k.op.(pw(k.k₁, x, x′), pw(k.k₂, x, x′))

# Unary operations.
ew(k::BinaryKernel, x::AV) = k.op.(ew(k.k₁, x), ew(k.k₂, x))
pw(k::BinaryKernel, x::AV) = k.op.(pw(k.k₁, x), pw(k.k₂, x))


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
ew(k::BinaryCrossKernel, x::AV, x′::AV) = k.op.(ew(k.k₁, x, x′), ew(k.k₂, x, x′))
pw(k::BinaryCrossKernel, x::AV, x′::AV) = k.op.(pw(k.k₁, x, x′), pw(k.k₂, x, x′))

# Unary operations.
ew(k::BinaryCrossKernel, x::AV) = k.op.(ew(k.k₁, x), ew(k.k₂, x))
pw(k::BinaryCrossKernel, x::AV) = k.op.(pw(k.k₁, x), pw(k.k₂, x))



############################## Multiply-by-Function Kernels ################################


"""
    LhsCross <: CrossKernel{Tf, Tk<:CrossKernel}

A cross-kernel given by `f(x) * k(x, x′)`.
"""
struct LhsCross{Tf<:MeanFunction, Tk<:CrossKernel} <: CrossKernel
    f::Tf
    k::Tk
end
ew(k::LhsCross, x::AV, x′::AV) = ew(k.f, x) .* ew(k.k, x, x′)
function pw(k::LhsCross, x::AV, x′::AV)
    return ew(k.f, x) .* pw(k.k, x, x′)
end
pw(k::LhsCross, x::AV) = pw(k, x, x)


"""
    RhsCross <: CrossKernel{Tk<:CrossKernel, Tf}

A cross-kernel given by `k(x, x′) * f(x′)`.
"""
struct RhsCross{Tk<:CrossKernel, Tf<:MeanFunction} <: CrossKernel
    k::Tk
    f::Tf
end
ew(k::RhsCross, x::AV, x′::AV) = ew(k.k, x, x′) .* ew(k.f, x′)
pw(k::RhsCross, x::AV, x′::AV) = pw(k.k, x, x′) .* ew(k.f, x′)'
pw(k::RhsCross, x::AV) = pw(k, x, x)

"""
    OuterCross <: CrossKernel{Tf, Tk<:CrossKernel}

A kernel given by `f(x) * k(x, x′) * f(x′)`.
"""
struct OuterCross{Tf<:MeanFunction, Tk<:CrossKernel} <: CrossKernel
    f::Tf
    k::Tk
end
ew(k::OuterCross, x::AV, x′::AV) = ew(k.f, x) .* ew(k.k, x, x′) .* ew(k.f, x′)
pw(k::OuterCross, x::AV, x′::AV) = ew(k.f, x) .* pw(k.k, x, x′) .* ew(k.f, x′)'


"""
    OuterKernel <: Kernel

A kernel given by `f(x) * k(x, x′) * f(x′)`.
"""
struct OuterKernel{Tf<:MeanFunction, Tk<:Kernel} <: Kernel
    f::Tf
    k::Tk
end

# Binary methods.
ew(k::OuterKernel, x::AV, x′::AV) = ew(k.f, x) .* ew(k.k, x, x′) .* ew(k.f, x′)
pw(k::OuterKernel, x::AV, x′::AV) = ew(k.f, x) .* pw(k.k, x, x′) .* ew(k.f, x′)'

# Unary methods.
function ew(k::OuterKernel, x::AV)
    fx = ew(k.f, x)
    return fx .* ew(k.k, x) .* fx
end
function pw(k::OuterKernel, x::AV)
    fx = ew(k.f, x)
    return fx .* pw(k.k, x) .* fx'
end
