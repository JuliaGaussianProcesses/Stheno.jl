using Flux, Flux.Tracker
using Flux.Tracker: track, @grad, TrackedVecOrMat, data

import Distances: pairwise

function pairwise(s::SqEuclidean, x::TrackedVecOrMat, x′::TrackedVecOrMat)
    return track(pairwise, s, x, x′)
end
function pairwise(s::SqEuclidean, x::TrackedVecOrMat, x′::AbstractVecOrMat)
    return track(pairwise, s, x, x′)
end
function pairwise(s::SqEuclidean, x::AbstractVecOrMat, x′::TrackedVecOrMat)
    return track(pairwise, s, x, x′)
end

@grad function pairwise(s::SqEuclidean, x::AbstractVecOrMat, x′::AbstractVecOrMat)
    D = pairwise(s, data(x), data(x′))
end
