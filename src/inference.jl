"""
    posterior()


"""
function posterior(d′::Vector{Normal})
    d = collect(keys(d′.gpc.obs))
    Kdd, Kd′d, Kd′d′ = cov(d), cov(d′, d), cov(d′, d′)

end
