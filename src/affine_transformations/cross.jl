#
# Functionality for constructing block matrices.
#

# Mangle name to avoid type piracy.
_mortar(blocks::AbstractArray) = BlockArrays.mortar(blocks)

function ChainRulesCore.rrule(::typeof(_mortar), _blocks::AbstractArray)
    mortar_pullback(Δ::Tangent) = (NoTangent(), Δ.blocks)
    return _mortar(_blocks), mortar_pullback
end

# A hook to which I can attach an rrule without commiting type-piracy against BlockArrays.
_collect(X::BlockArray) = Array(X)

function ChainRulesCore.rrule(::typeof(_collect), X::BlockArray)
    function Array_pullback(Δ::AbstractArray)
        ΔX = Tangent{Any}(blocks=BlockArray(collect(Δ), axes(X)).blocks, axes=NoTangent())
        return (NoTangent(), ΔX)
    end
    return Array(X), Array_pullback
end



#
# Functionality for grouping together collections of GPs into a single GP.
#

"""
    cross(fs::AbstractVector{<:AbstractGP})

Creates a GPPP-like object from a collection of GPs.
This is largely an implementation detail that is useful for GPPPs.
Not included in the user-facing API.
"""
function LinearAlgebra.cross(fs::AbstractVector{<:AbstractGP})
    consistency_checks(fs)
    return DerivedGP((cross, fs), first(fs).gpc)
end

function consistency_checks(fs)
    @assert length(fs) >= 1
    @assert all([f.gpc == first(fs).gpc for f in fs])
end
ChainRulesCore.@non_differentiable consistency_checks(::Any)

const cross_args{T<:AbstractVector{<:AbstractGP}} = Tuple{typeof(cross), T}

@opt_out rrule(::RuleConfig{>:HasReverseMode}, ::typeof(mean), ::cross_args, ::AV)
@opt_out rrule(::RuleConfig{>:HasReverseMode}, ::typeof(cov), ::cross_args, ::AV)
@opt_out rrule(::RuleConfig{>:HasReverseMode}, ::typeof(var), ::cross_args, ::AV)

function mean((_, fs)::cross_args, x::BlockData)
    blks = map((f, blk)->mean(f, blk), fs, blocks(x))
    return _collect(_mortar(blks))
end

function cov((_, fs)::cross_args, x::BlockData)
    Cs = reshape(map((f, blk)->cov(f, (cross, fs), blk, x), fs, blocks(x)), :, 1)
    return _collect(_mortar(reshape(Cs, :, 1)))
end

function var((_, fs)::cross_args, x::BlockData)
    cs = map(var, fs, blocks(x))
    return _collect(_mortar(cs))
end

function cov((_, fs)::cross_args, x::BlockData, x′::BlockData)
    Cs = reshape(map((f, blk)->cov(f, (cross, fs), blk, x′), fs, blocks(x)), :, 1)
    return _collect(_mortar(reshape(Cs, :, 1)))
end

function var((_, fs)::cross_args, x::BlockData, x′::BlockData)
    cs = map(var, fs, blocks(x), blocks(x′))
    return _collect(_mortar(cs))
end

function cov((_, fs)::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    Cs = reshape(map((f, x)->cov(f, f′, x, x′), fs, blocks(x)), :, 1)
    return _collect(_mortar(Cs))
end
function cov(f::AbstractGP, (_, fs)::cross_args, x::AV, x′::BlockData)
    Cs = reshape(map((f′, x′)->cov(f, f′, x, x′), fs, blocks(x′)), 1, :)
    return _collect(_mortar(Cs))
end

function var(args::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    return diag(cov(args, f′, x, x′))
end
function var(f::AbstractGP, args::cross_args, x::AV, x′::BlockData)
    return diag(cov(f, args, x, x′))
end
