"""
    differentiate(f::AbstractGP, differentiator)

Produces an AbstractGP `∂f` from `f` satisfying `∂ / ∂x f(x) == (∂f)(x)`.

`differentiator` must be a function...
"""
function differentiate(f::AbstractGP, differentiator)
    return CompositeGP((differentiate, f, differentiator), f.gpc)
end

const diff_gp = Tuple{typeof(differentiate), AbstractGP, Any}

function mean_vector((_, f, ∂)::diff_gp, x::AV{<:Real})
    return map(xn -> ∂(x -> only(mean_vector(f, [x])), xn), x)
end

cov(f::diff_gp, x::AV{<:Real}) = cov(f, x, x)
cov_diag(f::diff_gp, x::AV{<:Real}) = diag(cov(f, x))

function cov((_, f, ∂)::diff_gp, x::AV{<:Real}, x′::AV{<:Real})
    cols = map(x′) do x′n
        return map(x) do xn
            ∂(x′ -> ∂(x -> only(cov(f, [x], [x′])), xn), x′n)
        end
    end
    return hcat(cols...)
end
cov_diag(f::diff_gp, x::AV{<:Real}, x′::AV{<:Real}) = diag(cov(f, x, x′))

function cov((_, f, ∂)::diff_gp, f′::AbstractGP, x::AV{<:Real}, x′::AV{<:Real})
    cols = map(x′) do x′n
        return map(xn -> ∂(x -> only(cov(f, f′, [x], [x′n])), xn), x)
    end
    return hcat(cols...)
end
function cov_diag(f::diff_gp, f′::AbstractGP, x::AV{<:Real}, x′::AV{<:Real})
    return diag(cov(f, f′, x, x′))
end

function cov(f::AbstractGP, (_, f′, ∂)::diff_gp, x::AV{<:Real}, x′::AV{<:Real})
    cols = map(x′) do x′n
        return map(xn -> ∂(x′ -> only(cov(f, f′, [xn], [x′])), x′n), x)
    end
    return hcat(cols...)
end
function cov_diag(f::AbstractGP, f′::diff_gp, x::AV{<:Real}, x′::AV{<:Real})
    return diag(cov(f, f′, x, x′))
end
