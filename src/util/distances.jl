# Implement some extensions to Euclidean and Squared-Euclidean distances.
for (d, D) in [(:sqeuclidean, :SqEuclidean), (:euclidean, :Euclidean)]
    @eval begin
        ew(::$D, x::AV{<:Real}, x′::AV{<:Real}) = $d.(x, x′)
        pw(::$D, x::AV{<:Real}, x′::AV{<:Real}) = $d.(x, x′')

        ew(::$D, x::AV{T}) where {T<:Real} = zeros(T, length(x))
        pw(::$D, x::AV{<:Real}) = pairwise($D(), x, x)

        ew(::$D, x::ColVecs{<:Real}, x′::ColVecs{<:Real}) = colwise($D(), x.X, x′.X)
        pw(::$D, x::ColVecs{<:Real}, x′::ColVecs{<:Real}) = pw($D(), x.X, x′.X; dims=2)

        ew(::$D, x::ColVecs{T}) where {T<:Real} = zeros(T, length(x))
        pw(::$D, x::ColVecs{<:Real}) = pairwise($D(), x.X; dims=2)
    end
end

@adjoint function pairwise(::Euclidean, X::AV{<:Real})
    D, back = Zygote.forward(X->pairwise(SqEuclidean(), X), X)
    D .= sqrt.(D)
    return D, function(Δ)
        Δ = Δ ./ (2 .* D)
        Δ[diagind(Δ)] .= 0
        return (nothing, first(back(Δ)))
    end
end
