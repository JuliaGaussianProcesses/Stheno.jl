using Stheno: get_f, get_y, Observation, merge, GPC
using BlockArrays

function abs_rel_errs(x, y)
    δ = abs.(vec(x) .- vec(y))
    return [δ δ ./ vec(y)]
end

@testset "conditioning" begin

    @testset "Observation" begin
        rng, N, N′, D = MersenneTwister(123456), 5, 6, 2
        X, X′ = ColsAreObs(randn(rng, D, N)), ColsAreObs(randn(rng, D, N′))
        y, y′ = randn(rng, N), randn(rng, N′)
        f = GP(1, eq(), GPC())

        fX, fX′ = f(X), f(X′)
        c1, c2 = fX←y, fX′←y′
        @test Observation(fX, y) == c1
        @test get_f(c1) === fX && get_f(c2) === fX′
        @test get_y(c1) === y && get_y(c2) === y′

        c = merge((c1, c2))
        @test get_y(c) == BlockVector([y, y′])
        @test get_f(c).fs == [fX, fX′]
    end

    @testset "condition once" begin
        rng, N, N′, D = MersenneTwister(123456), 10, 3, 2
        x = collect(range(-3.0, stop=3.0, length=N))
        f = GP(1, eq(), GPC())
        y = rand(rng, f(x))

        # Test mechanics for finite conditioned process with single conditioning.
        f′ = f | (f(x, 1e-9)←y)
        @test maximum(abs.(rand(rng, f′(x, 1e-9)) - y)) < 1e-3
        @test maximum(abs.(mean(f′(x)) - y)) < 1e-3
        @test all(abs.(cov(f′(x))) .< 1e-6)
    end

    @testset "condition repeatedly" begin
        rng, N, N′ = MersenneTwister(123456), 5, 7
        xx′ = collect(range(-3.0, stop=3.0, length=N+N′))
        idx = rand(1:length(xx′), N)
        idx_1, idx_2 = idx, setdiff(1:length(xx′), idx)
        x, x′ = xx′[idx_1], xx′[idx_2]

        f = GP(1, eq(), GPC())
        y = rand(rng, f(xx′))
        y1, y2 = y[idx_1], y[idx_2]

        # Construct posterior using one conditioning operation.
        f′ = f | (f(xx′, eps()) ← y)

        # Construct posterior using two conditioning operations.
        f′1 = f | (f(x, eps()) ← y1)
        f′2 = f′1 | (f′(x′, eps()) ← y2)

        @test_broken mean(f′(xx′)) ≈ mean(f′2(xx′)) atol=1e-9 rtol=1e-9
        @test_broken cov(f′(xx′)) ≈ cov(f′2(xx′)) atol=1e-9 rtol=1e-9
        @test cov(f′(x), f′(x′)) ≈ cov(f′2(x), f′2(x′)) atol=1e-12 rtol=1e-12

        # # Check that each method is self-consistent.
        # Σ′XX′ = cov(f′XX′)
        # @test Σ′XX′ ≈ cov(f′(XX′))
        # @test Σ′XX′[1:N, 1:N] ≈ cov(f′X)[1:N, 1:N]
        # @test Σ′XX′[N+1:N+N′, N+1:N+N′] ≈ cov(f′X′)[1:N′, 1:N′]
        # # @test maximum(abs.(Σ′XX′[1:N, N+1:N+N′] - cov(f′X, f′X′, 1:N, 1:N′))) < 1e-12
        # # @test maximum(abs.(Σ′XX′[N+1:N+N′, 1:N] - cov(f′X′, f′X, 1:N′, 1:N))) < 1e-12
    end

    @testset "BlockGP" begin
        rng, N, N′ = MersenneTwister(123456), 5, 7
        xx′ = collect(range(-3.0, stop=3.0, length=N+N′))
        idx = rand(1:length(xx′), N)
        idx_1, idx_2 = idx, setdiff(1:length(xx′), idx)
        x, x′ = xx′[idx_1], xx′[idx_2]

        # Test that conditioning works for BlockGPs.
        fb, xb = BlockGP([f, f]), BlockData([x, x′])
        f′ = f | (fb(Xb)←BlockArray(y, [N, N′]))
        @test mean(fb′(Zb)) ≈ mean(f′(Zb))
        @test maximum(abs.(cov(fb′(Zb)) - cov(f′(Zb)))) < 1e-6

        # f′b = f | (fb(Xb)←BlockArray(y, [N, N′]))
        # @test mean(f′b(X)) ≈ mean(f′X)
        # @test maximum(abs.(cov(f′b(Zb)) - cov(f′(Zb)))) < 1e-6

        # # Test sugar for multiple-conditioning.
        # @test mean(fb′(Zb)) ≈ mean((f | (f(X)←y[1:N], f(X′)←y[N+1:end]))(Zb))
        # @test maximum(abs.(cov(fb′(Zb)) - cov((f | (f(X)←y[1:N], f(X′)←y[N+1:end]))(Zb)))) < 1e-6

        # yX = rand(rng, f(X))
        # f′X, f′X′ = (f(X), f(X′)) | (f(X)←yX)
        # f′X1, f′X′1 = f(X) | (f(X)←yX), f(X′) | (f(X)←yX)
        # @test mean(f′X) ≈ mean(f′X1)
        # @test mean(f′X′) ≈ mean(f′X′1)
        # @test maximum(abs.(cov(f′X) - cov(f′X1))) < 1e-6
        # @test maximum(abs.(cov(f′X′) - cov(f′X′1))) < 1e-6

        # yX, yX′ = y[1:N], y[N+1:end]
        # f′X, f′X′ = (f(X), f(X′)) | (f(X)←yX, f(X′)←yX′)
        # f′X1, f′X′1 = f(X) | (f(X)←yX, f(X′)←yX′), f(X′) | (f(X)←yX, f(X′)←yX′)
        # @test mean(f′X) ≈ mean(f′X1)
        # @test mean(f′X′) ≈ mean(f′X′1)
        # @test maximum(abs.(cov(f′X) - cov(f′X1))) < 1e-6
        # @test maximum(abs.(cov(f′X′) - cov(f′X′1))) < 1e-6
    end
end
