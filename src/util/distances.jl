# Implement some extensions to Euclidean and Squared-Euclidean distances.
for (d, D) in [(:sqeuclidean, :SqEuclidean), (:euclidean, :Euclidean)]
    @eval begin
        function ew(d::$D, x::AV{<:Real}, x′::AV{<:Real})
            return colwise($D(dtol), reshape(x, 1, :), reshape(x′, 1, :))
        end
        function pw(d::$D, x::AV{<:Real}, x′::AV{<:Real})
            return pw($D(dtol), reshape(x, 1, :), reshape(x′, 1, :); dims=2)
        end

        ew(::$D, x::AV{T}) where {T<:Real} = zeros(T, length(x))
        pw(::$D, x::AV{<:Real}) = pairwise($D(dtol), x, x)

        ew(::$D, x::ColVecs{<:Real}, x′::ColVecs{<:Real}) = colwise($D(dtol), x.X, x′.X)
        pw(::$D, x::ColVecs{<:Real}, x′::ColVecs{<:Real}) = pw($D(dtol), x.X, x′.X; dims=2)

        ew(::$D, x::ColVecs{T}) where {T<:Real} = zeros(T, length(x))
        pw(::$D, x::ColVecs{<:Real}) = pairwise($D(dtol), x.X; dims=2)
    end
end

function rrule(::typeof(pairwise), ::Euclidean, x::AbstractVector{<:Real})
    D, back = Zygote.pullback(x->pairwise(SqEuclidean(dtol), x), x)
    D .= sqrt.(D)
    return D, function(Δ)
        Δ = Δ ./ (2 .* max.(D, eps(eltype(D))))
        # Δ = Δ ./ (2 .* D)
        Δ[diagind(Δ)] .= 0
        return (NO_FIELDS, NO_FIELDS, first(back(Δ)))
    end
end
