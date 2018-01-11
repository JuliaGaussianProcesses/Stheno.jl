@testset "integrate" begin
    let
        f = GP(x->0.0, EQ(), GPC())
        v = ∫(f)
        @test kernel(v)(1, 1) == 1 / sqrt(3)
        @test kernel(v, f)(1, 5.0) == kernel(f, v)(5.0, 1)
        @test kernel(v, f)(1, 0.0) == 1 / sqrt(2)
    end
end
