import Base.BLAS: trsv
import PDMats: AbstractPDMat, invquad, dim
import Base: cov, logdet, full, size, chol, ==
export cov, invquad, AbstractPDMat

"""
    StridedPDMatrix

A strided positive definite matrix, represented in terms of it's Cholesky factorization `U`,
which should be an upper triangular matrix.
"""
struct StridedPDMatrix{T<:Real} <: AbstractPDMat{T}
    U::UpperTriangular{T}
end

dim(Σ::StridedPDMatrix) = size(Σ.U, 1)
full(Σ::StridedPDMatrix) = Σ.U'Σ.U
logdet(Σ::StridedPDMatrix) = 2 * sum(log, view(Σ.U, diagind(Σ.U)))
invquad(Σ::StridedPDMatrix, x::AbstractVector) = sum(abs2, trsv('U', 'T', 'N', Σ.U.data, x))
chol(Σ::StridedPDMatrix) = Σ.U
==(Σ1::StridedPDMatrix, Σ2::StridedPDMatrix) = Σ1.U == Σ2.U

"""
    cov(k::Kernel, x::T, y::T) where T<:Union{AbstractVector, RowVector}

Allocate memory for the covariance matrix and call `cov!`.
"""
function cov(k, x::AbstractVector)
    x1, x2 = reshape(x, length(x), 1), reshape(x, 1, length(x))
    return StridedPDMatrix(chol(Symmetric(k.(x1, x2) + 1e-12I)))
end
cov(k, x::RowVector) = StridedPDMatrix(chol(Symmetric(k.(x.vec, x)) + 1e-12I))
cov(k, x::T, y::T) where T<:RowVector = k.(x.vec, y)
cov(k, x::T, y::T) where T<:AbstractVector = k.(x, RowVector(y))

"""
    

Compute the covariance matrix implied by a collection of GPs. This should comprise a
collection of Tuples, the first element of which are GPs, the second input locations.
"""
function cov(obs::Vararg{Tuple{GP, T} where T<:Union{AbstractVector, RowVector}})
    N_p = length(obs)
    N = map(x->length(x[2]), obs)
    N_tot = sum(N)
    K = Matrix{Float64}(N_tot, N_tot)
    pos = 1
    for q in 1:N_p
        gp_q, x_q = obs[q][1], obs[q][2]
        for c in 1:N[q]
            for p in 1:N_p
                gp_p, x_p = obs[p][1], obs[p][2]
                k = kernel(gp_q, gp_p)
                for r in 1:N[p]
                    K[pos] = k(x_p[r], x_q[c])
                    pos +=1
                end
            end
        end
    end
    return StridedPDMatrix(chol(Symmetric(K) + 1e-12I))
end
