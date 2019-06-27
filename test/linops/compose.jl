using Stheno: GPC, Stretch, LinearTransform

@testset "compose" begin
    rng, N, N′, gpc = MersenneTwister(123456), 7, 5, GPC()
    x, x′ = randn(rng, N), randn(rng, N′)
    f, g, h = GP(sin, eq(), gpc), cos, GP(exp, linear(), gpc)
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

    @testset "Standardised Tests" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:l=>0.5, :σ=>2.3),
            θ->begin
                f = θ[:σ] * GP(sin, eq(l=θ[:l]), GPC())
                return f ∘ 0.5, f
            end,
            N, N′,
        )
    end
end
