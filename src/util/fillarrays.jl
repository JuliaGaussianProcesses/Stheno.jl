import LinearAlgebra: HermOrSym, diag, Diagonal

# Make getting the Diagonal of a Symmetric matrix that contains a Diagonal matrix efficient.
diag(S::Symmetric{T, <:Diagonal{T}} where T) = S.data.diag
diag(D::Diagonal{T, <:Fill{T, 1}} where T) = D.diag
Zygote._symmetric_back(Δ::Diagonal) = Δ

# constant-diagonal Diagonal matrix is closed under cholesky (ish)
function cholesky(A::Diagonal{T, <:Fill{T, 1}} where T)
    return Cholesky(Diagonal(Fill(sqrt(getindex_value(A.diag)), length(A.diag))), :U, 0)
end
@adjoint function cholesky(A::Diagonal{T, <:Fill{T, 1}} where T)
    return cholesky(A), function(Δ)
        d = sum(diag(Δ.factors)) / length(A.diag)
        return (Diagonal(Fill(d / (2 * sqrt(getindex_value(A.diag))), length(A.diag))),)
    end
end

# Diagonal matrices are always symmetric...
cholesky(A::HermOrSym{T, <:Diagonal{T}} where T) = cholesky(Diagonal(diag(A)))
