# Mean function optimisations.
const ZM = ZeroMean{Float64}
zero(::Type{<:MeanFunction}, N::Int) = FiniteZeroMean(1:N)
zero(::Type{<:MeanFunction}) = ZeroMean{Float64}()
zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()

+(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c + μ′.c)
*(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c * μ′.c)

# Kernel optimisations.
const ZK = ZeroKernel{Float64}
zero(k::Kernel) = length(k) < Inf ? FiniteZeroKernel(eachindex(k)) : ZK()

# ConstantKernel-specific optimisations.
+(k::ConstantKernel, k′::ConstantKernel) = ConstantKernel(k.c + k′.c)
*(k::ConstantKernel, k′::ConstantKernel) = ConstantKernel(k.c * k′.c)

function zero(k::CrossKernel)
    if size(k, 1) < Inf && size(k, 2) < Inf
        return FiniteZeroCrossKernel(eachindex(k, 1), eachindex(k, 2))
    elseif size(k, 1) < Inf
        return LhsFiniteZeroCrossKernel(eachindex(k, 1))
    elseif size(k, 2) < Inf
        return RhsFiniteZeroCrossKernel(eachindex(k, 2))
    else
        return ZeroKernel{Float64}()
    end
end

+(x::FiniteZeroMean, x′::FiniteZeroMean) = zero(x)


function +(μ::MeanFunction, μ′::MeanFunction)
    @assert size(μ) == size(μ′)
    if iszero(μ)
        return μ′
    elseif iszero(μ′)
        return μ
    else
        return CompositeMean(+, μ, μ′)
    end
end
function *(μ::MeanFunction, μ′::MeanFunction)
    @assert size(μ) == size(μ′)
    return iszero(μ) || iszero(μ′) ? zero(μ) : CompositeMean(*, μ, μ′)
end


############################## Convenience functionality ##############################

import Base: +, *, promote_rule, convert

function *(f, μ::MeanFunction)
    return iszero(μ) ? μ : CompositeMean(f, μ)
end

function *(f, k::CrossKernel)
    return iszero(k) ? k : LhsCross(f, k)
end
function *(k::CrossKernel, f)
    return iszero(k) ? k : RhsCross(k, f)
end
function *(f, k::RhsCross)
    if k.k isa Kernel && f == k.f
        return OuterKernel(f, k.k)
    else
        return LhsCross(f, k)
    end
end
function *(k::LhsCross, f)
    if k.k isa Kernel && f == k.f
        return OuterKernel(f, k.k)
    else
        return RhsCross(k, f)
    end
end

promote_rule(::Type{<:MeanFunction}, ::Type{<:Real}) = MeanFunction
convert(::Type{MeanFunction}, x::Real) = ConstantMean(x)

promote_rule(::Type{<:Kernel}, ::Type{<:Real}) = Kernel
convert(::Type{<:CrossKernel}, x::Real) = ConstantKernel(x)

# Composing mean functions with Reals.
+(μ::MeanFunction, μ′::Real) = +(promote(μ, μ′)...)
+(μ::Real, μ′::MeanFunction) = +(promote(μ, μ′)...)

*(μ::MeanFunction, μ′::Real) = *(promote(μ, μ′)...)
*(μ::Real, μ′::MeanFunction) = *(promote(μ, μ′)...)

# Composing kernels with Reals.
+(k::CrossKernel, k′::Real) = +(promote(k, k′)...)
+(k::Real, k′::CrossKernel) = +(promote(k, k′)...)

*(k::CrossKernel, k′::Real) = *(promote(k, k′)...)
*(k::Real, k′::CrossKernel) = *(promote(k, k′)...)


import Base: zero, +, *

const ZM = ZeroMean{Float64}
zero(::Type{<:MeanFunction}, N::Int) = FiniteZeroMean(1:N)
zero(::Type{<:MeanFunction}) = ZeroMean{Float64}()
zero(μ::MeanFunction) = length(μ) < Inf ? FiniteZeroMean(eachindex(μ)) : ZM()

# Addition of mean functions.
function +(μ::MeanFunction, μ′::MeanFunction)
    if iszero(μ)
        return μ′
    elseif iszero(μ′)
        return μ
    else
        return BinaryMean(+, μ, μ′)
    end
end
+(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c + μ′.c)
+(a::Real, μ::MeanFunction) = UnaryMean(m->a + m, μ)
+(μ::MeanFunction, a::Real) = UnaryMean(m->m + a, μ)

# Product of mean functions.
function *(μ::MeanFunction, μ′::MeanFunction)
    if iszero(μ)
        return μ
    elseif iszero(μ′)
        return μ′
    else
        return BinaryMean(*, μ, μ′)
    end
end
*(μ::ConstantMean, μ′::ConstantMean) = ConstantMean(μ.c * μ′.c)
# *(a::Real, μ::MeanFunction) = UnaryMean(m->a * m, μ)
# *(μ::MeanFunction, a::Real) = UnaryMean(m->m * a, μ)

@inline *(a::Real, μ::MeanFunction) = BinaryMean(*, a, μ)
@inline *(μ::MeanFunction, a::Real) = BinaryMean(*, μ, a)

import Base: iszero, zero, *

# Define addition of zeros.
for T in [
    :FiniteZeroKernel,
    :LhsFiniteZeroCrossKernel,
    :RhsFiniteZeroCrossKernel,
    :FiniteZeroCrossKernel,
]
    @eval +(x::$T, x′::$T) = zero(x)
end

# Define zeros for block stuff.
zero(μ::BlockMean) = BlockMean(zero.(μ.μ))
zero(k::BlockCrossKernel) = BlockCrossKernel(zero.(k.ks))
zero(k::BlockKernel) = BlockKernel(zero.(k.ks_diag), zero.(k.ks_off))
