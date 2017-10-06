@testset "input_transform" begin

    # Test construction.
    @test InputTransformedKernel(EQ(), identity).k == EQ()

    # Test getter methods.
    let k = InputTransformedKernel(EQ(), identity)
        @test kernel(k) == EQ()
        @test input_transform(k) == identity
    end

    # Test equality.
    let
        k1 = InputTransformedKernel(EQ(), identity)
        k2 = InputTransformedKernel(RQ(1.0), identity)
        k3 = InputTransformedKernel(EQ(), sin)
        @test k1 == k1
        @test k2 == k2
        @test k3 == k3
        @test k1 != k2
        @test k1 != k3
        @test k2 != k3
    end

    # Test functionality under equality.
    let k = InputTransformedKernel(EQ(), identity)
        @test k(5.0, 4.0) == kernel(k)(5.0, 4.0)
    end

    # Test equivalence between InputTransformedKernel and just applying the function.
    let rng = MersenneTwister(123456)
        k1 = InputTransformedKernel(EQ(), sin)
        k2 = InputTransformedKernel(RQ(5.0), tanh)
        x1, x2 = randn.((rng, rng))
        @test k1(x1, x2) == EQ()(sin(x1), sin(x2))
        @test k2(x1, x2) == RQ(5.0)(tanh(x1), tanh(x2))
    end

    # Test that Index InputTransformedKernel works.
    let
        k1 = InputTransformedKernel(EQ(), Index{1}())
        k2 = InputTransformedKernel(EQ(), Index{2}())
        x1, x2 = [5.0, 4.0], [3.0, 2.0]
        @test EQ()(x1[1], x2[1]) == k1(x1, x2)
        @test EQ()(x1[2], x2[2]) == k2(x1, x2)

        x1t, x2t = (5.0, 4.0), (3.0, 2.0)
        @test k1(x1, x2) == k1(x1t, x2t)
        @test k2(x1, x2) == k2(x1t, x2t)

        @test k1(x1, x2) == Index{1}(EQ())(x1, x2)
        @test k2(x1, x2) == Index{2}(EQ())(x1, x2)
    end

end
