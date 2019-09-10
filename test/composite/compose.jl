using Stheno: GPC, EQ, Exp

@timedtestset "compose" begin
    rng, N, N′, gpc = MersenneTwister(123456), 5, 3, GPC()
    x, x′ = randn(rng, N), randn(rng, N′)
    f, g, h = GP(sin, EQ(), gpc), cos, GP(exp, Exp(), gpc)
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

    @timedtestset "Consistency Tests" begin
        P, Q = 4, 3
        x0, x1, x2, x3 = randn(rng, P), randn(rng, Q), randn(rng, Q), randn(rng, P)
        abstractgp_interface_tests(fg, f, x0, x1, x2, x3)
        abstractgp_interface_tests(stretch(f, 0.1), f, x0, x1, x2, x3)

        # f = GP(EQ(), GPC())
        # abstractgp_interface_tests(periodic(f, 0.1), f, x0, x1, x2, x3)
    end
    @timedtestset "Diff Tests" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:σ=>0.5),
            θ->begin
                f = θ[:σ] * GP(sin, EQ(), GPC())
                return stretch(f, 0.5), f
            end,
            collect(range(-2.0, 2.0; length=N)),
            collect(range(-1.5, 2.2; length=N′)),
        )
    end
end
