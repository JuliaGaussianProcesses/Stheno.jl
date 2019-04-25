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

function \(
    A::UpperTriangular{T, <:Diagonal{T}} where {T},
    X::AbstractVecOrMat,
)
    return A.data \ X
end
