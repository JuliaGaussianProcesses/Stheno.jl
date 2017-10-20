export posterior

"""
    posterior()


"""
function posterior(d′::Normal)

    # Get processes with observations and compute a concatenated vector of their means
    # and the observed values.
    d = convert(Vector{Normal}, collect(keys(d′.gpc.obs)))
    f, μ = getindex.(d′.gpc.obs, d), map(mean, d)
    f_cat = vcat(f...)
    μ_cat = vcat(map((d, μ)->μ.(1:dims(d)), d, μ)...)

    # Compute covariance matrices.
    println(typeof(d))
    println(typeof(d′))
    Σdd, Σd′d, Σd′d′ = cov(d), cov(d′, d), cov(d′, d′)
    println(size(f_cat))
    println(size(μ_cat))

    # Compute the posterior distribution and return it.
    U = chol(Σdd)
    println(U)
    μ_post = Σd′d * (U \ At_ldiv_B(U, f_cat .- μ_cat))
    Σ_post = Σd′d′ .- Σd′d * (U \ At_ldiv_B(U, permutedims(Σd′d, [2, 1])))
    return Normal(nothing, nothing, n->μ_post[n], (m, n)->Σ_post[m, n], dims(d′), GPC())
end
