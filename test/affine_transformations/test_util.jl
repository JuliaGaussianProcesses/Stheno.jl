function standard_1D_tests(rng::AbstractRNG, θ, f, x::AbstractVector, z::AbstractVector)
    g, u = f(θ)

    @test cov(g, x) ≈ cov(g, x)'
    @test minimum(eigvals(cov(g, x))) > -1e-9
    @test var(g, x) ≈ diag(cov(g, x))

    @test cov(g, g, x, x) ≈ cov(g, x)
    @test var(g, g, x, x) ≈ diag(cov(g, g, x, x))
end
