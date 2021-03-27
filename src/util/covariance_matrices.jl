# Various specialised operations using the Cholesky factorisation.

Xt_invA_X(A::Cholesky, x::AbstractVector) = sum(abs2, A.U' \ x)
function Xt_invA_X(A::Cholesky, X::AbstractVecOrMat)
    V = A.U' \ X
    return Symmetric(V'V)
end

Xt_invA_Y(X::AbstractVecOrMat, A::Cholesky, Y::AbstractVecOrMat) = (A.U' \ X)' * (A.U' \ Y)

diag_At_A(A::AbstractVecOrMat) = vec(sum(abs2.(A); dims=1))

function diag_At_B(A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B)
    return vec(sum(A .* B; dims=1))
end

diag_Xt_invA_X(A::Cholesky, X::AbstractVecOrMat) = diag_At_A(A.U' \ X)

function diag_Xt_invA_Y(X::AbstractMatrix, A::Cholesky, Y::AbstractMatrix)
    @assert size(X) == size(Y)
    return diag_At_B(A.U' \ X, A.U' \ Y)
end
