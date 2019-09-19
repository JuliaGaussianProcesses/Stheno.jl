using Stheno: GPC, EQ, Exp

@timedtestset "addition" begin
    @timedtestset "Correlated GPs" begin
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        f1, f2 = GP(1, EQ(), gpc), GP(2, Exp(), gpc)
        f3 = f1 + f2
        f4 = f1 + f3
        f5 = f3 + f4

        for (n, (fp, fa, fb)) in enumerate([(f3, f1, f2), (f4, f1, f3), (f5, f3, f4)])
            Σp = cov(fa(X)) + cov(fb(X)) + cov(fa(X), fb(X)) + cov(fb(X), fa(X))
            ΣpXX′ = cov(fa(X), fa(X′)) + cov(fb(X), fb(X′)) + cov(fa(X), fb(X′)) +
                cov(fa(X), fb(X′))
            @test mean(fp(X)) ≈ mean(fa(X)) + mean(fb(X))
            @test cov(fp(X)) ≈ Σp
            @test cov(fp(X), fp(X′)) ≈ ΣpXX′
            @test cov(fp(X′), fp(X)) ≈ transpose(ΣpXX′)                
            @test cov(fp(X), fa(X′)) ≈ cov(fa(X), fa(X′)) + cov(fb(X), fa(X′))
            @test cov(fp(X′), fa(X)) ≈ cov(fa(X′), fa(X)) + cov(fb(X′), fa(X))
            @test cov(fa(X), fp(X′)) ≈ cov(fb(X), fa(X′)) + cov(fa(X), fa(X′))
            @test cov(fa(X′), fp(X)) ≈ cov(fb(X′), fa(X)) + cov(fa(X′), fa(X))
        end

        @timedtestset "Consistency Tests" begin
            P, Q = 4, 3
            x0, x1, x2, x3 = randn(rng, P), randn(rng, Q), randn(rng, Q), randn(rng, P)
            abstractgp_interface_tests(f3, f1, x0, x1, x2, x3)
            abstractgp_interface_tests(f2 - f1, f1, x0, x1, x2, x3)
        end
    end
    @timedtestset "Verify mean / kernel numerically" begin
        rng, N, D = MersenneTwister(123456), 5, 6
        X = ColsAreObs(randn(rng, D, N))
        c, f = randn(rng), GP(5, EQ(), GPC())

        @test mean((f + c)(X)) == mean(f(X)) .+ c
        @test mean((f + c)(X)) == c .+ mean(f(X))
        @test cov((f + c)(X)) == cov(f(X))
        @test cov((c + f)(X)) == cov(f(X))

        @test mean((f - c)(X)) == mean(f(X)) .- c
        @test mean((c - f)(X)) == c .- mean(f(X))
        @test cov((f - c)(X)) == cov(f(X))
        @test cov((c - f)(X)) == cov(f(X))

        x = randn(rng, N + D)
        @test mean((f + sin)(x)) == mean(f(x)) + map(sin, x)
        @test mean((sin + f)(x)) == map(sin, x) + mean(f(x))
        @test cov((f + sin)(x)) == cov(f(x))
        @test cov((sin + f)(x)) == cov(f(x))

        @timedtestset "Consistency Tests" begin
            P, Q = 5, 3
            x0, x1, x2, x3 = randn(rng, P), randn(rng, Q), randn(rng, Q), randn(rng, P)
            abstractgp_interface_tests(c + f, f, x0, x1, x2, x3)
        end
    end
    @timedtestset "Standardised Tests (independent sum)" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:l1=>0.5, :l2=>2.3),
            θ->begin
                gpc = GPC()
                f1 = θ[:l1] * GP(sin, EQ(), gpc)
                f2 = θ[:l2] * GP(cos, EQ(), gpc)
                f3 = f1 + f2
                return f3, f3
            end,
            collect(range(-1.5, 1.5; length=5)),
            collect(range(-1.0, 0.5; length=3)),
        )
    end
    @timedtestset "Standardised Tests (correlated sum)" begin
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:l1=>0.5, :l2=>2.3),
            θ->begin
                gpc = GPC()
                f1 = θ[:l1] * GP(sin, EQ(), gpc)
                f2 = θ[:l2] * GP(cos, EQ(), gpc)
                f3 = f1 + f2
                f4 = f1 + f3
                f5 = f3 + f4
                return f5, f5
            end,
            collect(range(-1.5, 1.5; length=5)),
            collect(range(-1.0, 0.5; length=3)),
        )
    end
end

# # θ = Dict(:l1=>0.5, :l2=>2.3);
# x, z = collect(range(-5.0, 5.0; length=512)), collect(range(-5.0, 5.0; length=128));
# y = rand(GP(sin, eq(), GPC())(x, 0.1));

# foo_logpdf = (x, y) -> begin
#     gpc = GPC()
#     f = GP(sin, eq(), gpc)
#     return logpdf(f(x, 0.1), y)
# end

# foo_elbo = (x, y, z) -> begin
#     f = GP(0, eq(), GPC())
#     return elbo(f(x, 0.1), y, f(z, 0.001))
# end

# @benchmark foo_logpdf($x,  $y)
# @benchmark Zygote.forward(foo_logpdf, $x, $y)

# let
#     z, back = Zygote.forward(foo_logpdf, x, y)
#     @benchmark $back($(randn()))
# end

# let
#     foo = function(x, y)
#         fx = GP(0, eq(), GPC())(x, 0.1)
#         C = cholesky(Symmetric(cov(fx)))
#         return logdet(C) + Xt_invA_X(C, y)
#     end
#     display(@benchmark Zygote.forward($foo, $x, $y)) 
#     z_pw, back_pw = Zygote.forward(foo, x, y)
#     @benchmark $back_pw(randn())
# end


# @benchmark foo_elbo($x, $y, $z)
# @benchmark Zygote.forward(foo_elbo, $x, $y, $z)

# let
#     L, back = Zygote.forward(foo_elbo, x, y, z)
#     @benchmark $back($L)
# end


# θ->begin
#     gpc = GPC()
#     f1 = GP(sin, eq(l=θ[:l1]), gpc)
#     f2 = GP(cos, eq(l=θ[:l2]), gpc)
#     f3 = f1 + f2
#     return f3, f3
# end,
# 13, 11,
