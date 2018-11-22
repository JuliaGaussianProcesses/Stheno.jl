using Flux, Flux.Tracker
using Flux.Tracker: track, @grad, TrackedVecOrMat, data

import Distances: pairwise
function pairwise(s::SqEuclidean, X::TrackedVecOrMat, Y::TrackedVecOrMat)
    return track(pairwise, s, X, Y)
end
function pairwise(s::SqEuclidean, X::TrackedVecOrMat, Y::AbstractVecOrMat)
    return track(pairwise, s, X, Y)
end
function pairwise(s::SqEuclidean, X::AbstractVecOrMat, Y::TrackedVecOrMat)
    return track(pairwise, s, X, Y)
end
@grad function pairwise(s::SqEuclidean, x::AbstractVecOrMat, y::AbstractVecOrMat)
    D = pairwise(s, data(x), data(y))
    return D, function(Δ)
        x̄ = -2 * y * Δ' + 2 * x * Diagonal(sum(Δ; dims=2))
        ȳ = -2 * x * Δ  + 2 * y * Diagonal(sum(Δ; dims=1))
        return x̄, ȳ
    end
end
