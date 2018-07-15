export project

"""
    project(ϕ, f::GP, g)

The projection operation `f′(x) = Σₘ ϕₘ(x) f(m) + g(x)`. Assumes `f` is finite.
"""
function project(ϕ, f::GP, g)
    @assert isfinite(length(f))
    return GP(project, ϕ, f, g)
end

function project(ϕ, f::GP)
    if size(ϕ, 2) < Inf
        return project(ϕ, f, FiniteZeroMean(eachindex(ϕ, 2)))
    else
        return project(ϕ, f, ZeroMean{Float64}())
    end
end

function μ_p′(::typeof(project), ϕ, f, g)
    return iszero(mean(f)) ? g : DeltaSumMean(ϕ, mean(f), g)
end
function k_p′(::typeof(project), ϕ, f, g)
    if iszero(kernel(f))
        return get_zero(size(ϕ, 2), size(ϕ, 2))
    else
        return DeltaSumKernel(ϕ, kernel(f))
    end
end
function k_p′p(::typeof(project), ϕ, f, g, fp)
    if iszero(kernel(f, fp))
        return get_zero(size(ϕ, 2), length(fp))
    else
        return LhsDeltaSumCrossKernel(ϕ, kernel(f, fp))
    end
end
function k_pp′(fp, ::typeof(project), ϕ, f, g)
    if iszero(kernel(f, fp))
        return get_zero(length(fp), size(ϕ, 2))
    else
        return RhsDeltaSumCrossKernel(kernel(f, fp), ϕ)
    end
end
