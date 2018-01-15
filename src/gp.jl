import Base: mean, show, cov, chol, eachindex, transpose
export mean, mean_vector, kernel, GP, GPC, condition!, predict, lpdf, sample, dims

const __ϵ = 1e-9

# A collection of GPs (GPC == "GP Collection"). Primarily used to track cross-kernels.
struct GPC
    gps::Set{Any}
    k_x::ObjectIdDict
    GPC() = new(Set{Any}(), ObjectIdDict())
end

"""
    GP{Tμ<:μFun, Tk<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tμ<:μFun, Tk<:Kernel}
    f::Any
    args::Any
    μ::Tμ
    k::Tk
    gpc::GPC
    GP{Tμ, Tk}(f, args, μ, k::Tk, gpc::GPC) where {Tμ<:μFun, Tk<:Kernel} =
        new{Tμ, Tk}(f, args, μ, k, gpc)
    function GP{Tμ, Tk}(μ::Tμ, k::Tk, gpc::GPC) where {Tμ<:μFun, Tk<:Kernel}
        gp = new(GP, nothing, μ, k, gpc)
        push!(gpc.gps, gp)
        return gp
    end
end
GP(μ::Tμ, k::Tk, gpc::GPC) where {Tμ<:μFun, Tk<:Kernel} = GP{Tμ, Tk}(μ, k, gpc)
function GP(op, args...)
    μ, k, gpc = μ_p′(op, args...), k_p′(op, args...), get_check_gpc(op, args...)
    new_gp = GP{typeof(μ), typeof(k)}(op, args, μ, k, gpc)
    for gp in gpc.gps
        gpc.k_x[(new_gp, gp)] = k_p′p(gp, op, args...)
        gpc.k_x[(gp, new_gp)] = k_pp′(gp, op, args...)
    end
    push!(gpc.gps, new_gp)
    return new_gp
end
const SthenoType = Union{Real, Function, GP}
show(io::IO, gp::GP) = print(io, "GP with μ = ($(gp.μ)) k=($(gp.k)) f=($(gp.f))")
transpose(f::GP) = f
isfinite(f::GP) = isfinite(f.k)

@inline dims(d::GP) = dims(kernel(d))
@inline eachindex(f::GP) = 1:dims(f)

mean(z::Real) = ConstantMean(z)
mean(f::Function) = CustomMean(f)
mean(f::GP) = f.μ
mean_vector(f::GP) = mean(f).(eachindex(f))
mean_vector(f::Vector{<:GP}) = vcat(mean_vector.(f)...)

"""
    kernel(f::Union{Real, Function})
    kernel(f::GP)
    kernel(f::Union{Real, Function}, g::GP)
    kernel(f::GP, g::Union{Real, Function})
    kernel(fa::GP, fb::GP)

Get the cross-kernel between `GP`s `fa` and `fb`, and . If either argument is deterministic
then the zero-kernel is returned.
`kernel(f) === kernel(f, f)`
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    return fa === fb ?
        fa.k :
        (fa, fb) in keys(fa.gpc.k_x) ? fa.gpc.k_x[(fa, fb)] : Constant(0.0)
end
kernel(::Union{Real, Function}) = Zero()
kernel(::Union{Real, Function}, ::GP) = Zero()
kernel(::GP, ::Union{Real, Function}) = Zero()

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

"""
    cov(d::Union{GP, Vector{<:GP}}, d′::Union{GP, Vector{<:GP}})

Compute the cross-covariance between GPs (or vectors of) `d` and `d′`.
"""
cov(d::Vector{<:GP}, d′::Vector{<:GP}) = cov(kernel.(d, RowVector(d′)))
cov(d::Vector{<:GP}, d′::GP) = cov(d, [d′])
cov(d::GP, d′::Vector{<:GP}) = cov([d], d′)
cov(d::GP, d′::GP) = cov([d], [d′])

"""
    cov(d::Union{GP, Vector{<:GP}})

Compute the marginal covariance matrix for GP (or vector thereof) `d`.
"""
function cov(d::Vector{<:GP})
    K = cov(kernel.(d, RowVector(d)))::Matrix{Float64}
    K[diagind(K)] .+= __ϵ
    LAPACK.potrf!('U', K)
    return StridedPDMatrix(UpperTriangular(K))
end
cov(d::GP) = cov([d])
