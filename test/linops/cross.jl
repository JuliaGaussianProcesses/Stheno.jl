@testset "cross" begin
    rng, P, Q, gpc = MersenneTwister(123456), 11, 13, GPC()

    f1 = GP(sin, EQ(), gpc)
    f2 = GP(cos, EQ(), gpc)
    f3 = cross([f1, f2])

    # x, z = 

end
