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
