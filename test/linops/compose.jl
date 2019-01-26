using Stheno: GPC

@testset "compose" begin
    rng, N, N′, gpc = MersenneTwister(123456), 7, 5, GPC()
    x, x′ = randn(rng, N), randn(rng, N′)
    f, g, h = GP(sin, EQ(), gpc), cos, GP(exp, Linear(), gpc)
    fg = f ∘ g

    # Check marginals statistics inductively.
    @test mean(fg(x)) == mean(f(map(g, x)))
    @test cov(fg(x)) == cov(f(map(g, x)))
    
    # Check cross covariance between transformed process and original inductively.
    @test cov(fg(x), fg(x′)) == cov(f(map(g, x)), f(map(g, x′)))
    @test cov(fg(x), f(x′)) == cov(f(map(g, x)), f(x′))
    @test cov(f(x), fg(x′)) == cov(f(x), f(map(g, x′)))

    # Check cross covariance between transformed process and independent absolutely.
    @test cov(fg(x), h(x′)) == zeros(length(x), length(x′))
    @test cov(h(x), fg(x′)) == zeros(length(x), length(x′))
end
