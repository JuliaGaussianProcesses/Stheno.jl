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
