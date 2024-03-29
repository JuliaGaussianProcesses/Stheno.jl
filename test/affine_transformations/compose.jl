@timedtestset "compose" begin
    @timedtestset "general" begin
        rng, N, N′, gpc = MersenneTwister(123456), 5, 3, GPC()
        x, x′ = randn(rng, N), randn(rng, N′)
        f = atomic(GP(sin, SEKernel()), gpc)
        g = cos
        h = atomic(GP(exp, ExponentialKernel()), gpc)
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

            # f = GP(SqExponentialKernel(), GPC())
            # abstractgp_interface_tests(periodic(f, 0.1), f, x0, x1, x2, x3)
        end
        @timedtestset "Diff Tests" begin
            standard_1D_tests(
                MersenneTwister(123456),
                Dict(:σ=>0.5),
                θ->begin
                    f = θ[:σ] * atomic(GP(sin, SEKernel()), GPC())
                    return stretch(f, 0.5), f
                end,
                collect(range(-2.0, 2.0; length=N)),
                collect(range(-1.5, 2.2; length=N′)),
            )
        end
    end
    @timedtestset "Stretch" begin
        @timedtestset "scalar stretch" begin
            rng, N, λ = MersenneTwister(123456), 3, 0.51
            x = randn(rng)
            f = atomic(GP(1.3, SEKernel()), GPC())
            g = stretch(f, λ)

            @timedtestset "scalar input" begin
                @test first(cov(f, g, [0.0], [0.0])) == 1.0
                @test first(cov(f, g, [λ * x], [x])) == 1.0
                standard_1D_tests(
                    MersenneTwister(123456),
                    Dict(:σ=>0.5, :l=>0.32),
                    θ->begin
                        f_ = θ[:σ] * atomic(GP(sin, SEKernel()), GPC())
                        return stretch(f_, θ[:l]), f_
                    end,
                    collect(range(-2.0, 2.0; length=N)),
                    collect(range(-1.5, 2.2; length=5)),
                )
                @testset "StepRangeLen" begin
                    v = range(-2.0, 2.0; length=N)
                    @test cov(g(v)) ≈ cov(g(collect(v)))
                end
            end
            @timedtestset "vector input" begin
                D = 11
                X = randn(rng, D, 1)
                @test first(cov(f, g, ColVecs(zeros(D, 1)), ColVecs(zeros(D, 1)))) == 1.0
                @test first(cov(f, g, ColVecs(λ .* X), ColVecs(X))) == 1.0
            end
        end
        @timedtestset "Vector stretch" begin
            rng, N, D = MersenneTwister(123456), 3, 7
            λ = randn(rng, D)
            f = atomic(GP(1.0, SEKernel()), GPC())
            g = stretch(f, λ)

            X = randn(rng, D, 1)
            @test first(cov(f, g, ColVecs(zeros(D, 1)), ColVecs(zeros(D, 1)))) == 1.0
            @test first(cov(f, g, ColVecs(Diagonal(λ) * X), ColVecs(X))) == 1.0
        end
        @timedtestset "Matrix stretch" begin
            rng, N, D = MersenneTwister(123456), 3, 7
            A = randn(rng, D, D)
            f = atomic(GP(1.0, SEKernel()), GPC())
            g = stretch(f, A)

            X = randn(rng, D, 1)
            @test first(cov(f, g, ColVecs(zeros(D, 1)), ColVecs(zeros(D, 1)))) == 1.0
            @test first(cov(f, g, ColVecs(A * X), ColVecs(X))) == 1.0
        end
    end
    @timedtestset "Select" begin
        rng, N, D = MersenneTwister(123456), 3, 6

        @timedtestset "evaluation" begin
            t = Stheno.Select(D - 1)
            x = ColVecs(randn(rng, D, N))
            @test t.(x) == [t(_x) for _x in x]

            t = Stheno.Select((D-1):D)
            @test t.(x) == [t(_x) for _x in x]
        end
        @timedtestset "idx isa Integer" begin
            idx = 1
            f = atomic(GP(1.3, SEKernel()), GPC())
            g = select(f, idx)

            X = randn(rng, D, N)
            X_f = X[idx, :]
            X_g = ColVecs(X)
            @test cov(f, g, X_f, X_g) ≈ cov(f, X_f, X_f)
            @test cov(f, g, X_f, X_g) ≈ cov(g, X_g, X_g)
        end
        @timedtestset "idx isa Vector" begin
            idx = [1, 3]
            f = atomic(GP(1.3, SEKernel()), GPC())
            g = select(f, idx)

            X = randn(rng, D, N)
            X_f = ColVecs(X[idx, :])
            X_g = ColVecs(X)
            @test cov(f, g, X_f, X_g) ≈ cov(f, X_f, X_f)
            @test cov(f, g, X_f, X_g) ≈ cov(g, X_g, X_g)
        end
    end
    @timedtestset "Shift" begin
        @timedtestset "Shift{Float64}" begin
            rng, N, D = MersenneTwister(123456), 3, 6
            a = randn(rng)
            f = atomic(GP(1.3, SEKernel()), GPC())
            g = shift(f, a)

            x = randn(rng, N)
            x_f = x .- a
            x_g = x
            @test cov(f, g, x_f, x_g) ≈ cov(f, x_f, x_f)
            @test cov(f, g, x_f, x_g) ≈ cov(g, x_g, x_g)

            X = randn(rng, D, N)
            X_f = ColVecs(X .- a)
            X_g = ColVecs(X)
            @test cov(f, g, X_f, X_g) ≈ cov(f, X_f, X_f)
            @test cov(f, g, X_f, X_g) ≈ cov(g, X_g, X_g)
        end
        @timedtestset "Shift{Vector{Float64}}" begin
            rng, N, D = MersenneTwister(123456), 3, 6
            a = randn(rng, D)
            f = atomic(GP(1.3, SEKernel()), GPC())
            g = shift(f, a)

            X = randn(rng, D, N)
            X_f = ColVecs(X .- a)
            X_g = ColVecs(X)
            @test cov(f, g, X_f, X_g) ≈ cov(f, X_f, X_f)
            @test cov(f, g, X_f, X_g) ≈ cov(g, X_g, X_g)
        end
    end
    @timedtestset "Periodic" begin
        f = atomic(GP(SEKernel()), GPC())
        g = periodic(f, 2.0)
        @test cov(g([0.0]), g([1.0])) ≈ cov(g([0.0]), g([3.0]))

        t = Stheno.Periodic(2.0)
        x = randn(5)
        @test t.(x) == [t(_x) for _x in x]
    end
end
