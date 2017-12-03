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
    GP{Tk<:Kernel}

A Gaussian Process (GP) object. Either constructed using an Affine Transformation of
existing GPs or by providing a mean function `μ`, a kernel `k`, and a `GPC` `gpc`.
"""
struct GP{Tk<:Kernel}
    f::Any
    args::Any
    μ::Any
    k::Tk
    gpc::GPC
    GP{Tk}(f, args, μ, k::Tk, gpc::GPC) where Tk<:Kernel = new{Tk}(f, args, μ, k, gpc)
    function GP{Tk}(μ, k::Tk, gpc::GPC) where Tk<:Kernel
        gp = new(GP, nothing, μ, k, gpc)
        push!(gpc.gps, gp)
        return gp
    end
end
function GP(op, args...)
    gpc = get_check_gpc(op, args...)
    k = k_p′(op, args...)
    new_gp = GP{typeof(k)}(op, args, μ_p′(op, args...), k, gpc)
    for gp in gpc.gps
        gpc.k_x[(new_gp, gp)] = k_p′p(gp, op, args...)
        gpc.k_x[(gp, new_gp)] = k_pp′(gp, op, args...)
    end
    push!(gpc.gps, new_gp)
    return new_gp
end
GP(μ, k::Tk, gpc::GPC) where Tk = GP{Tk}(μ, k, gpc)
show(io::IO, gp::GP) = print(io, "GP with μ = ($(gp.μ)) k=($(gp.k)) f=($(gp.f))")
transpose(f::GP) = f
isfinite(f::GP) = isfinite(f.k)

@inline dims(d::GP) = dims(kernel(d))
@inline eachindex(f::GP) = 1:dims(f)

mean(f::GP) = f.μ
mean_vector(f::GP) = mean(f).(eachindex(f))
mean_vector(f::Vector) = vcat(mean_vector.(f)...)

"""
    kernel(f::GP)
    kernel(fa::GP, fb::GP)

Get the cross-kernel between `GP`s `fa` and `fb`, and `kernel(f) == kernel(f, f)`.
"""
kernel(f::GP) = f.k
function kernel(fa::GP, fb::GP)
    @assert fa.gpc === fb.gpc
    return fa === fb ?
        fa.k :
        (fa, fb) in keys(fa.gpc.k_x) ? fa.gpc.k_x[(fa, fb)] : Constant(0.0)
end

function get_check_gpc(args...)
    gpc = args[findfirst(map(arg->arg isa GP, args))].gpc
    @assert all([!(arg isa GP) || arg.gpc == gpc for arg in args])
    return gpc
end

"""
    cov(d::Union{GP, Vector{GP}}, d′::Union{GP, Vector{GP}})

Compute the cross-covariance between GPs (or vectors of) `d` and `d′`.
"""
cov(d::Vector{GP}, d′::Vector{GP}) = cov(kernel.(d, RowVector(d′)))
cov(d::Vector{GP}, d′::GP) = cov(d, Vector{GP}([d′]))
cov(d::GP, d′::Vector{GP}) = cov(Vector{GP}([d]), d′)
cov(d::GP, d′::GP) = cov(Vector{GP}([d]), Vector{GP}([d′]))

"""
    cov(d::Union{GP, Vector{GP}})

Compute the marginal covariance matrix for GP (or vector thereof) `d`.
"""
cov(d::Vector{GP}) = StridedPDMatrix(chol(cov(kernel.(d, RowVector(d))) + __ϵ * I))
cov(d::GP) = cov(Vector{GP}([d]))
