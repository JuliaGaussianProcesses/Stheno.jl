import LinearAlgebra: cholesky, HermOrSym, diag, Diagonal

# Implementation for `Diagonal` sensitivities as they don't work property atm.
@adjoint function Diagonal(d::AbstractVector)
    back(Δ::NamedTuple) = (Δ.diag,)
    back(Δ::AbstractMatrix) = (diag(Δ),)
    return Diagonal(d), back
end

# Make getting the Diagonal of a Symmetric matrix that contains a Diagonal matrix efficient.
diag(S::Symmetric{T, <:Diagonal{T}} where T) = S.data.diag
diag(D::Diagonal{T, <:Fill{T, 1}} where T) = D.diag
Zygote._symmetric_back(Δ::Diagonal) = Δ

# constant-diagonal Diagonal matrix is closed under cholesky (ish)
function cholesky(A::Diagonal{T, <:Fill{T, 1}} where T)
    return Cholesky(Diagonal(Fill(sqrt(getindex_value(A.diag)), length(A.diag))), :U, 0)
end
@adjoint function cholesky(A::Diagonal{T, <:Fill{T, 1}} where T)
    return cholesky(A), Δ->(Diagonal(Fill(1 / (2 * sqrt(getindex_value(A.diag))), length(A.diag))),)
end

# Diagonal matrices are always symmetric...
cholesky(A::HermOrSym{T, <:Diagonal{T}} where T) = cholesky(Diagonal(diag(A)))
