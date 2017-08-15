module Stheno

# Module-wide definitions for consistency.
export CovMat, AVector
const CovMat = Matrix{Float64}
const AVector = Union{AbstractVector, RowVector}

include("cov.jl")

end # module
