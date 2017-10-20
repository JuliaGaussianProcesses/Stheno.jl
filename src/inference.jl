export posterior

"""
    posterior()


"""
function posterior(d′::Vector{Normal})

    # Get processes with observations and compute a concatenated vector of their means
    # and the observed values.
    d = convert(Vector{Normal}, collect(keys(d′.gpc.obs)))
    f, μ = getindex.(d′.gpc.obs, d), map(mean, d)
    f_cat, μ_cat = vcat(f...), vcat(map((d, μ)->μ.(1:dims(d)), d, μ)...)

    # Compute covariance matrices.
    Σdd, Σd′d, Σd′d′ = cov(d), cov(d′, d), cov(d′, d′)

    # Compute the posterior distribution and return it.
    U = chol(Σdd)
    μ_post = Σd′d * (U \ At_ldiv_B(U, f_cat .- μ_cat)) .+ mean(d′).(1:dims(d′))
    Σ_post = Σd′d′ .- Σd′d * (U \ At_ldiv_B(U, permutedims(Σd′d, [2, 1])))
    return Normal(nothing, nothing, n->μ_post[n], (m, n)->Σ_post[m, n], dims(d′), GPC())
end
posterior(d′::Normal) = posterior([d′])
