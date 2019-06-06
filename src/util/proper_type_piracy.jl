# This file contains some real first-rate type piracy. Sorry in advance.

import LinearAlgebra: \

function \(
    A::Adjoint{T, <:UpperTriangular{T, <:Diagonal{T}}} where {T},
    X::AbstractMatrix,
)
    return A.parent.data \ X
end

function \(
    A::Adjoint{T, <:UpperTriangular{T, <:Diagonal{T}}} where {T},
    X::AbstractVector,
)
    return A.parent.data \ X
end

\(A::UpperTriangular{T, <:Diagonal{T}} where {T}, x::AbstractVector) = A.data \ x
\(A::UpperTriangular{T, <:Diagonal{T}} where {T}, X::AbstractMatrix) = A.data \ X
