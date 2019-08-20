import Base: ∘
export ∘, select, stretch, periodic

"""
    ∘(f::GP, g)

Constructs the GP f′ given by f′(x) := f(g(x))
"""
∘(f::AbstractGP, g) = CompositeGP((∘, f, g), f.gpc)


const comp_args = Tuple{typeof(∘), AbstractGP, Any}

mean_vector((_, f, g)::comp_args, x::AV) = mean_vector(f, g.(x))

cov((_, f, g)::comp_args, x::AV) = cov(f, g.(x))
cov_diag((_, f, g)::comp_args, x::AV) = cov_diag(f, g.(x))

cov((_, f, g)::comp_args, x::AV, x′::AV) = cov(f, g.(x), g.(x′))
cov_diag((_, f, g)::comp_args, x::AV, x′::AV) = cov_diag(f, g.(x), g.(x′))

cov((_, f, g)::comp_args, f′::AbstractGP, x::AV, x′::AV) = cov(f, f′, g.(x), x′)
cov(f::AbstractGP, (_, f′, g)::comp_args, x::AV, x′::AV) = cov(f, f′, x, g.(x′))

cov_diag((_, f, g)::comp_args, f′::AbstractGP, x::AV, x′::AV) = cov_diag(f, f′, g.(x), x′)
cov_diag(f::AbstractGP, (_, f′, g)::comp_args, x::AV, x′::AV) = cov_diag(f, f′, x, g.(x′))


"""
    Stretch{T<:Real}

Stretch all elements of the inputs by `l`.
"""
struct Stretch{T<:Real}
    l::T
end
(s::Stretch)(x) = s.l * x
broadcasted(s::Stretch{<:Real}, x::StepRangeLen) = s.l .* x
broadcasted(s::Stretch{<:Real}, x::ColsAreObs) = ColsAreObs(s.l .* x.X)
broadcasted(s::Stretch{<:AbstractMatrix}, x::ColsAreObs) = ColsAreObs(s.l * x.X)

"""
    stretch(f::AbstractGP, l::Union{AbstractVecOrMat, Real})

Equivalent to `f ∘ Stretch(l)`
"""
stretch(f::AbstractGP, l::Real) = f ∘ Stretch(l)
stretch(f::AbstractGP, a::AbstractVector) = stretch(f, Diagonal(a))
stretch(f::AbstractGP, A::AbstractMatrix) = f ∘ LinearTransform(A)



"""
    Select{Tidx}

Use inside an input transformation meanfunction / crosskernel to improve peformance.
"""
struct Select{Tidx}
    idx::Tidx
end
(f::Select)(x) = x[f.idx]
broadcasted(f::Select, x::ColsAreObs) = ColsAreObs(x.X[f.idx, :])
broadcasted(f::Select{<:Integer}, x::ColsAreObs) = x.X[f.idx, :]

function broadcasted(f::Select, x::AbstractVector{<:CartesianIndex})
    out = Matrix{Int}(undef, length(f.idx), length(x))
    for i in f.idx, n in eachindex(x)
        out[i, n] = x[n][i]
    end
    return ColsAreObs(out)
end
@adjoint function broadcasted(f::Select, x::AV{<:CartesianIndex})
    return broadcasted(f, x), Δ->(nothing, nothing)
end
function broadcasted(f::Select{<:Integer}, x::AV{<:CartesianIndex})
    out = Matrix{Int}(undef, length(x))
    for n in eachindex(x)
        out[n] = x[n][f.idx]
    end
    return out
end

"""
    select(f::AbstractGP, idx)

Select the dimensions of the input to `f` given by `idx`.
"""
select(f::AbstractGP, idx) = f ∘ Select(idx)



"""
    Periodic{Tf<:Real}

Make a kernel or mean function periodic by projecting into two dimensions.
"""
struct Periodic{Tf<:Real}
    f::Tf
end
(p::Periodic)(t::Real) = [cos((2π * p.f) * t), sin((2π * p.f) * t)]
function broadcasted(p::Periodic, x::AbstractVector{<:Real})
    return ColsAreObs(vcat(cos.((2π * p.f) .* x)', sin.((2π * p.f) .* x)'))
end

"""
    periodic(g::AbstractGP, f::Real)

Produce an AbstractGP with period `f`.
"""
periodic(g::AbstractGP, f::Real) = g ∘ Periodic(f)
