"""
    cross(fs::AbstractVector{<:GP})

Creates a multi-output GP from an `AbstractVector` of `GP`s.
"""
function LinearAlgebra.cross(fs::AbstractVector{<:AbstractGP})
    consistency_checks(fs)
    return CompositeGP((cross, fs), first(fs).gpc)
end

function consistency_checks(fs)
    @assert length(fs) >= 1
    @assert all([f.gpc == first(fs).gpc for f in fs])
end
ChainRulesCore.@non_differentiable consistency_checks(::Any)


const cross_args{T<:AbstractVector{<:AbstractGP}} = Tuple{typeof(cross), T}

function mean((_, fs)::cross_args, x::BlockData)
    blks = map((f, blk)->mean(f, blk), fs, blocks(x))
    return _collect(mortar(blks))
end

function cov((_, fs)::cross_args, x::BlockData)
    Cs = reshape(map((f, blk)->cov(f, (cross, fs), blk, x), fs, blocks(x)), :, 1)
    return _collect(mortar(reshape(Cs, :, 1)))
end

function var((_, fs)::cross_args, x::BlockData)
    cs = map(var, fs, blocks(x))
    return _collect(mortar(cs))
end

function cov((_, fs)::cross_args, x::BlockData, x′::BlockData)
    Cs = reshape(map((f, blk)->cov(f, (cross, fs), blk, x′), fs, blocks(x)), :, 1)
    return _collect(mortar(reshape(Cs, :, 1)))
end

function var((_, fs)::cross_args, x::BlockData, x′::BlockData)
    cs = map(var, fs, blocks(x), blocks(x′))
    return _collect(mortar(cs))
end

function cov((_, fs)::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    Cs = reshape(map((f, x)->cov(f, f′, x, x′), fs, blocks(x)), :, 1)
    return _collect(mortar(Cs))
end
function cov(f::AbstractGP, (_, fs)::cross_args, x::AV, x′::BlockData)
    Cs = reshape(map((f′, x′)->cov(f, f′, x, x′), fs, blocks(x′)), 1, :)
    return _collect(mortar(Cs))
end

function var(args::cross_args, f′::AbstractGP, x::BlockData, x′::AV)
    return diag(cov(args, f′, x, x′))
end
function var(f::AbstractGP, args::cross_args, x::AV, x′::BlockData)
    return diag(cov(f, args, x, x′))
end



#
# Util for multi-process versions of `rand`, `logpdf`, and `elbo`.
#

function finites_to_block(fs::AV{<:FiniteGP})
    return FiniteGP(
        cross(map(f->f.f, fs)),
        BlockData(map(f->f.x, fs)),
        make_block_noise(map(f->f.Σy, fs)),
    )
end

make_block_noise(Σys::Vector{<:Diagonal}) = Diagonal(Vector(mortar(diag.(Σys))))
make_block_noise(Σys::Vector) = block_diagonal(Σys)
