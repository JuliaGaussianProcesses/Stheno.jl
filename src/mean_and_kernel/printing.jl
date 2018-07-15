import Base: print

# Base means.
print(io::IO, m::EmpiricalMean) = print(io, "EmpiricalMean")

# Base kenrels.
print(io::IO, k::ZeroKernel) = print(io, "0")
print(io::IO, k::ConstantKernel) = print(io, k.c)
print(io::IO, k::EmpiricalKernel) = print(io, "EmpiricalKernel")

# Composite Kernels.
const Composite{T} = Union{CompositeMean{T}, CompositeKernel{T}, CompositeCrossKernel{T}}
function print(io::IO, shunted::Shunted{T}) where T<:Composite
    k, shunt = shunted.x, shunted.shunt
    println(io, string(shunt) * "$T")
    print_shunted_list(io, extend(shunt), k.x)
end
print(io::IO, x::Composite) = print(io, dummy_shunt(x))

const Destructurable = Union{LhsCross, RhsCross, OuterCross, OuterKernel,
                            ITMean, ITKernel,
                            ConditionalMean, ConditionalKernel, ConditionalCrossKernel,
                            FiniteMean, FiniteKernel, FiniteCrossKernel,
                            LhsFiniteCrossKernel, RhsFiniteCrossKernel,
                            DeltaSumMean, DeltaSumKernel,
                            LhsDeltaSumCrossKernel, RhsDeltaSumCrossKernel}
function print(io::IO, shunted::Shunted{T}) where T<:Destructurable
    k, shunt = shunted.x, shunted.shunt
    println(io, string(shunt) * "$T")
    print_shunted_list(io, extend(shunt), destructure_fields(k))
end
print(io::IO, k::Destructurable) = print(io, dummy_shunt(k))

destructure_fields(k::Union{LhsCross, OuterCross, OuterKernel}) = (k.f, k.k)
destructure_fields(k::RhsCross) = (k.k, k.f)
destructure_fields(k::ITMean) = (k.μ, k.f)
destructure_fields(k::ITKernel) = (k.k, k.f)
destructure_fields(μ::ConditionalMean) = (μ.μg, μ.kfg)
destructure_fields(k::ConditionalKernel) = (k.kfg, k.kgg)
destructure_fields(k::ConditionalCrossKernel) = (k.kfg, k.kfh, k.kgh)
destructure_fields(μ::FiniteMean) = (μ.μ,)
destructure_fields(
    k::Union{
        FiniteKernel,
        FiniteCrossKernel,
        LhsFiniteCrossKernel,
        RhsFiniteCrossKernel,
    },
) = (k.k,)
destructure_fields(μ::DeltaSumMean) = (μ.ϕ, μ.μ, μ.g)
destructure_fields(
    k::Union{
        DeltaSumKernel,
        LhsDeltaSumCrossKernel,
        RhsDeltaSumCrossKernel,
    },
) = (k.ϕ, k.k)

# Block stuff.
function print(io::IO, shunted::Shunted{T}) where T<:BlockMean
    μ, shunt = shunted.x, shunted.shunt
    println(io, string(shunt) * "$T")
    for (j, x) in enumerate(μ.μ)
        println(io, Shunted(extend(shunt, 2), "$j: "))
        print(io, Shunted(extend(shunt), x))
        j != length(μ.μ) && print('\n')
    end
end
print(io::IO, μ::BlockMean) = print(io, dummy_shunt(μ))

function print(io::IO, shunted::Shunted{T}) where T<:BlockCrossKernel
    k, shunt = shunted.x, shunted.shunt
    println(io, string(shunt) * "$T")
    for p in 1:size(k.ks, 1), q in 1:size(k.ks, 2)
        println(io, Shunted(extend(shunt, 2), "($q, $p): "))
        print(io, Shunted(extend(shunt), k.ks[p, q]))
        (q * p) != length(k.ks) && print('\n')
    end
end
function print(io::IO, shunted::Shunted{T}) where T<:BlockKernel
    k, shunt = shunted.x, shunted.shunt
    println(io, string(shunt) * "$T")
    for p in 1:size(k.ks_off, 1), q in 1:size(k.ks_off, 2)
        println(io, Shunted(extend(shunt, 2), "($q, $p): "))
        if p != q
            print(io, Shunted(extend(shunt), k.ks_off[p, q]))
        else
            print(io, Shunted(extend(shunt), k.ks_diag[p]))
        end
        (q * p) != length(k.ks_off) && print('\n')
    end
end
print(io::IO, μ::Union{BlockKernel, BlockCrossKernel}) = print(io, dummy_shunt(μ))
