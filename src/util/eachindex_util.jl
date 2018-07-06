import Base: eachindex

eachindex(X::AbstractMatrix, dim::Int) = 1:size(X, dim)
function eachindex(X::AbstractBlockMatrix, dim::Int)
    if dim == 1
        return BlockVector(eachindex.(X.blocks[:, 1], dim))
    elseif dim == 2
        return BlockVector(eachindex.(X.blocks[1, :], dim))
    else
        throw(ErrorException("Bad dimension."))
    end
end
eachindex(X::BS, dim::Int) = eachindex(X.data, dim)
eachindex(X::BlockTri, dim::Int) = eachindex(X.data, dim)
eachindex(X::LazyPDMat, dim::Int) = eachindex(X.Σ, dim)
