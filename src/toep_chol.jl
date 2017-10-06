function toeplitz_chol(μ::Vector{<:Real})
    N = (length(μ) - 1) / 2
    τ = zeros()

end

C = Circulant([0.0, 1.0, 2.0, 3.0, 4.0])
μ_circ = (C.'C + 1e-6I)[:, 1]
μ = Toeplitz(μ_circ, μ_circ)
