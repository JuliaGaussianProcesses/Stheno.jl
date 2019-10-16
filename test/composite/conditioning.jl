using Stheno: get_f, get_y, Observation, merge, GPC
using BlockArrays, LinearAlgebra

function abs_rel_errs(x, y)
    δ = abs.(vec(x) .- vec(y))
    return [δ δ ./ vec(y)]
end

@timedtestset "conditioning" begin
    @timedtestset "Observation" begin
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        X, X′ = ColVecs(randn(rng, D, N)), ColVecs(randn(rng, D, N′))
        y, y′ = randn(rng, N), randn(rng, N′)
        f = GP(1, EQ(), GPC())

        fX, fX′ = f(X), f(X′)
        c1, c2 = fX←y, fX′←y′
        @test Observation(fX, y) == c1
        @test get_f(c1) === fX && get_f(c2) === fX′
        @test get_y(c1) === y && get_y(c2) === y′

        c = merge((c1, c2))
        @test get_y(c) == BlockVector([y, y′])
        @test get_y(Obs((c1, c2))) == BlockVector([y, y′])
    end
    @timedtestset "condition once" begin
        rng, N, N′, D = MersenneTwister(123456), 10, 3, 2
        x = collect(range(-3.0, stop=3.0, length=N))
        f = GP(1, EQ(), GPC())
        y = rand(rng, f(x))

        # Test mechanics for finite conditioned process with single conditioning.
        f′ = f | (f(x, 1e-9)←y)
        @test maximum(abs.(rand(rng, f′(x, 1e-9)) - y)) < 1e-3
        @test maximum(abs.(mean(f′(x)) - y)) < 1e-3
        @test all(abs.(cov(f′(x))) .< 1e-6)
    end
    @timedtestset "condition repeatedly" begin
        rng, N, N′ = MersenneTwister(123456), 5, 7
        xx′ = collect(range(-3.0, stop=3.0, length=N+N′))
        idx = randperm(rng, length(xx′))[1:N]
        idx_1, idx_2 = idx, setdiff(1:length(xx′), idx)
        x, x′ = xx′[idx_1], xx′[idx_2]

        f = GP(1, EQ(), GPC())
        y = rand(rng, f(xx′))
        y1, y2 = y[idx_1], y[idx_2]

        # Construct posterior using one conditioning operation.
        f′ = f | (f(xx′, 0.1) ← y)

        # Construct posterior using two conditioning operations.
        f′1 = f | (f(x, 0.1) ← y1)
        f′2 = f′1 | (f′1(x′, 0.1) ← y2)

        # Check that conditioning twice yields the same answer.
        @test mean(f′(xx′)) ≈ mean(f′2(xx′))
        @test cov(f′(xx′)) ≈ cov(f′2(xx′))
        @test cov(f′(x), f′(x′)) ≈ cov(f′2(x), f′2(x′))
    end
    @timedtestset "Consistency Tests" begin
        rng, N, P, Q = MersenneTwister(123456), 11, 5, 3
        f = GP(sin, EQ(), GPC())
        x = collect(range(-1.0, 1.0; length=N))
        y = randn(rng, N)
        f′ = f | (f(x) ← y)
        x0, x1, x2, x3 = randn(rng, P), randn(rng, Q), randn(rng, Q), randn(rng, P)
        abstractgp_interface_tests(f′, f, x0, x1, x2, x3)

        # Multiple things to condition on
        P′ = 7
        x′ = randn(rng, P′)
        y, y′ = rand(rng, [f(x, 1e-9), f(x′, 1e-9)])
        f′ = f | (f(x, 1e-9)←y, f(x′, 1e-9)←y′)
        abstractgp_interface_tests(f′, f, x0, x1, x2, x3)

        # Condition multiple things
        f′, f′′ = (f, f) | (f(x, 1e-9)←y)
        abstractgp_interface_tests(f′, f′′, x0, x1, x2, x3)

        # Condition multiple things on multiple things
        f′, f′′ = (f, f) | (f(x, 1e-9)←y, f(x′, 1e-9)←y′)
        abstractgp_interface_tests(f′, f′′, x0, x1, x2, x3)
    end
    @timedtestset "Diff Tests" begin
        rng, N, P, Q = MersenneTwister(123456), 11, 5, 3
        x_obs, A_obs = collect(range(-5.0, 5.0; length=N)), randn(rng, N, N)
        y_obs = rand(rng, GP(sin, EQ(), GPC())(x_obs, _to_psd(A_obs)))
        standard_1D_tests(
            MersenneTwister(123456),
            Dict(:x=>x_obs, :y=>y_obs, :A=>A_obs),
            θ->begin
                f = GP(sin, EQ(), GPC())
                f′ = f | (f(θ[:x], _to_psd(θ[:A]))←θ[:y])
                return f′, f′
            end,
            collect(range(-4.5, 4.5; length=P)),
            collect(range(-4.0, 5.0; length=Q)),
        )
    end
    @timedtestset "Conditioning with cross" begin
        rng, N, N′, σ² = MersenneTwister(123456), 3, 7, 1.0
        xx′ = collect(range(-3.0, stop=3.0, length=N+N′))
        xp = collect(range(-4.0, stop=4.0, length=N+N′+10))
        xp′ = collect(range(-4.0, stop=4.0, length=N+N′+11))
        f = GP(sin, EQ(), GPC())
        yy′ = rand(rng, f(xx′, σ²))

        # Chop up into blocks.
        idx = randperm(rng, length(xx′))[1:N]
        idx_1, idx_2 = idx, setdiff(1:length(xx′), idx)
        x, x′ = xx′[idx_1], xx′[idx_2]
        y, y′ = yy′[idx_1], yy′[idx_2]

        f′ = f | (f(xx′, σ²)←yy′)
        fb′ = f | (cross([f, f])(BlockData([x, x′]), σ²)←vcat(y, y′))
        fmc′ = f | (f(x, σ²)←y, f(x′, σ²)←y′)
        f′_obs = f | Obs(f(x, σ²)←y, f(x′, σ²)←y′)

        @test mean(f′(xp)) ≈ mean(fb′(xp))
        @test mean(f′(xp)) ≈ mean(fmc′(xp))
        @test mean(f′(xp)) ≈ mean(f′_obs(xp))

        @test cov(f′(xp)) ≈ cov(fb′(xp))
        @test cov(f′(xp)) ≈ cov(fmc′(xp))
        @test cov(f′(xp)) ≈ cov(f′_obs(xp))

        @test cov(f′(xp), f′(xp)) ≈ cov(fb′(xp), fb′(xp))
        @test cov(f′(xp), f′(xp)) ≈ cov(fmc′(xp), fmc′(xp))
        @test cov(f′(xp), f′(xp)) ≈ cov(f′_obs(xp), f′_obs(xp))

        @test cov(f′(xp), f′(xp′)) ≈ cov(fb′(xp), fb′(xp′))
        @test cov(f′(xp), f′(xp′)) ≈ cov(fmc′(xp), fmc′(xp′))
        @test cov(f′(xp), f′(xp′)) ≈ cov(f′_obs(xp), f′_obs(xp′))
    end
end
