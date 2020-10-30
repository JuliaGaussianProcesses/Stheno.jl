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

function cov(_, f, ∂)::diff_gp, x::AV{<:Real})

end
function cov_diag(_, f, ∂)::diff_gp, x::AV{<:Real})

end

function cov((_, f, ∂)::diff_gp, x::AV{<:Real}, x′::AV{<:Real})

end
function cov_diag((_, f, ∂)::diff_gp, x::AV{<:Real}, x′::AV{<:Real})

end

function cov((_, f, ∂)::diff_gp, f′::AbstractGP, x::AV{<:Real}, x′::AV{<:Real})

end
function cov_diag((_, f, ∂)::diff_gp, f′::AbstractGP, x::AV{<:Real}, x′::AV{<:Real})

end

function cov(f::AbstractGP, (_, f′, ∂)::diff_gp, x::AV{<:Real}, x′::AV{<:Real})

end
function cov_diag(f::AbstractGP, (_, f′, ∂)::diff_gp, x::AV{<:Real}, x′::AV{<:Real})

end
