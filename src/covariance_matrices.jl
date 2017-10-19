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
    cov(d::Normal)

The covariance matrix of a finite dimensional Normal distribution.
"""
cov(d::Normal) = cov(kernel(d), 1:dims(d))

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
    cov(obs::Vector{Tuple{GP, T}}) where T<:Union{AbstractVector, RowVector}

Compute the covariance matrix implied by a collection of GPs. This should comprise a
collection of Tuples, the first element of which are GPs, the second input locations.
"""
function cov(d::Vector{Normal})
    N = map(dims, d)
    K = Matrix{Float64}(sum(N), sum(N))
    pos = 1
    for q in eachindex(d)
        for c in 1:N[q]
            for p in eachindex(d)
                k = kernel(d[p], d[q])
                for r in 1:N[p]
                    K[pos] = k(r, c)
                    pos +=1
                end
            end
        end
    end
    return StridedPDMatrix(chol(Symmetric(K) + 1e-12I))
end

"""
    cov(d::Vector{Normal}, d′::Vector{Normal})

Compute the cross-covariance between the 
"""
function cov(d::Vector{Normal}, d′::Vector{Normal})
    Dx, Dx′ = map(dims, d), map(dims, d′)
    K = Matrix{Float64}(sum(Dx), sum(Dx′))
    pos = 1
    for q in eachindex(d′)
        for c in 1:Dx′[q]
            for p in eachindex(d)
                k = kernel(d[p], d′[q])
                for r in 1:Dx[p]
                    K[pos] = k(r, c)
                    pos += 1
                end
            end
        end
    end
    return K
end
