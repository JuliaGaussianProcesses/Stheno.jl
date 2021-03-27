# Various specialised operations using the Cholesky factorisation.

Xt_invA_X(A::Cholesky, x::AbstractVector) = sum(abs2, A.U' \ x)
function Xt_invA_X(A::Cholesky, X::AVM)
    V = A.U' \ X
    return Symmetric(V'V)
end

Xt_invA_Y(X::AVM, A::Cholesky, Y::AVM) = (A.U' \ X)' * (A.U' \ Y)

diag_At_A(A::AVM) = vec(sum(abs2.(A); dims=1))

function diag_At_B(A::AVM, B::AVM)
    @assert size(A) == size(B)
    return vec(sum(A .* B; dims=1))
end

diag_Xt_invA_X(A::Cholesky, X::AVM) = diag_At_A(A.U' \ X)

function diag_Xt_invA_Y(X::AM, A::Cholesky, Y::AM)
    @assert size(X) == size(Y)
    return diag_At_B(A.U' \ X, A.U' \ Y)
end
