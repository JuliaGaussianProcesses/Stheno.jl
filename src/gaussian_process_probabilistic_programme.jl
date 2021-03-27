"""
    

"""
struct GaussianProcessProbabilisticProgramme{Tfs<:Dict{Int}} <: AbstractGP
    fs::Tfs
    gpc::GPC
end

const GPPP = GaussianProcessProbabilisticProgramme

"""


"""
struct GPPPInput{Tp<:Integer, T, Tx<:AbstractVector{T}} <: AbstractVector{Tuple{Tp, T}}
    p::Tp
    x::Tx
end

Base.size(x::GPPPInput) = (length(x.x), )

Base.getindex(x::GPPPInput, idx) = map(x_ -> (x.p, x_), x.x[idx])



# As a first-pass, I'm letting the backend involve integer arithmetic, an planning to build
# helper functionality on top.

AbstractGPs.mean(f::GPPP, x::GPPPInput) = mean(f.fs[x.p], x.x)

AbstractGPs.cov(f::GPPP, x::GPPPInput) = cov(f.fs[x.p], x.x)

AbstractGPs.cov_diag(f::GPPP, x::GPPPInput) = cov_diag(f.fs[x.p], x.x)

function AbstractGPs.cov(f::GPPP, x::GPPPInput, x′::GPPPInput)
    return cov(f.fs[x.p], f.fs[x′.p], x.x, x′.x)
end

function AbstractGPs.cov_diag(f::GPPP, x::GPPPInput, x′::GPPPInput)
    return cov_diag(f.fs[x.p], f.fs[x′.p], x.x, x′.x)
end

AbstractGPs.mean_and_cov(f::GPPP, x::GPPPInput) = mean_and_cov(f.fs[x.p], x.x)

AbstractGPs.mean_and_cov_diag(f::GPPP, x::GPPPInput) = mean_and_cov_diag(f.fs[x.p], x.x)
