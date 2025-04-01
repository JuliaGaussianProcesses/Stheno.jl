@testset "addition" begin
    @testset "Correlated GPs" begin
        rng, N, N′, D, gpc = MersenneTwister(123456), 5, 6, 2, GPC()
        X, X′ = ColVecs(randn(rng, D, N)), ColVecs(randn(rng, D, N′))
        f1 = atomic(GP(1, SEKernel()), gpc)
        f2 = atomic(GP(2, SEKernel()), gpc)
        f3 = f1 + f2
        f4 = f1 + f3
        f5 = f3 + f4

        for (n, (fp, fa, fb)) in enumerate([(f3, f1, f2), (f4, f1, f3), (f5, f3, f4)])
            Σp = cov(fa(X)) + cov(fb(X)) + cov(fa(X), fb(X)) + cov(fb(X), fa(X))
            ΣpXX′ =
                cov(fa(X), fa(X′)) +
                cov(fb(X), fb(X′)) +
                cov(fa(X), fb(X′)) +
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

        @testset "Consistency Tests" begin
            P, Q = 4, 3
            x0, x1, x2, x3 = randn(rng, P), randn(rng, Q), randn(rng, Q), randn(rng, P)
            abstractgp_interface_tests(f3, f1, x0, x1, x2, x3)
            abstractgp_interface_tests(f2 - f1, f1, x0, x1, x2, x3)
        end
    end
    @testset "Verify mean / kernel numerically" begin
        rng, N, D = MersenneTwister(123456), 5, 6
        X = ColVecs(randn(rng, D, N))
        c, f = randn(rng), atomic(GP(5, SEKernel()), GPC())

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

        @testset "Consistency Tests" begin
            P, Q = 5, 3
            x0, x1, x2, x3 = randn(rng, P), randn(rng, Q), randn(rng, Q), randn(rng, P)
            abstractgp_interface_tests(c + f, f, x0, x1, x2, x3)
        end
    end
end
