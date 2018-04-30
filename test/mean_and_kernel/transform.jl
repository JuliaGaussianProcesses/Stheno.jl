# @testset "transform" begin

#     # Tests for Circularised Kernels.
#     @test lb(Circularised(EQ(), 0.0, 1.0)) == 0.0
#     @test ub(Circularised(EQ(), 0.0, 1.0)) == 1.0
#     @test Circularised(EQ(), 0.0, 1.0) == Circularised(EQ(), 0.0, 1.0)
#     @test Circularised(RQ(1.0), 0.0, 1.0) != Circularised(EQ(), 0.0, 1.0)
#     @test Circularised(EQ(), 0.5, 1.0) != Circularised(EQ(), 0.0, 1.0)
#     @test Circularised(EQ(), 0.0, 0.5) != Circularised(EQ(), 0.0, 1.0)
#     @test Circularised(EQ(), 0.0, 1.0)(0.0, 0.0) == 1.0
#     @test Circularised(EQ(), 0.0, 1.0)(0.0, 0.0) == EQ()(0.0, 0.0)

#     # Test some invariances.
#     let k = Circularised(EQ(), 0.0, 1.0)
#         @test k(0.0, 0.2) ≈ k(0.8, 1.0)
#         @test k(0.2, 0.0) ≈ k(0.0, 0.2)
#         @test k(0.0, 0.2) ≈ k(1.0, 0.8)
#         @test k(0.0, 1e-5) ≈ k(1.0 - 1e-5, 1.0)
#     end
# end
