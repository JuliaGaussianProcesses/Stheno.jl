using Stheno: Parameter, value
using Bijectors
using Random

rng = MersenneTwister(123456)

@testset "parameter wrapper" begin
    @testset "bijector" begin
        # scalar
        l = rand(rng)
        # vector
        N = 5
        v = rand(rng, N)
        # matrix
        M, M′ = 4, 7
        W = rand(rng, M, M′)

        # identity bijector
        pl = Parameter(l)
        @test l == value(pl)
        pv = Parameter(v)
        @test v == value(pv)
        pW = Parameter(W)
        @test W == value(pW)

        # Exp/Log bijector
        pl_pos = Parameter(l, Val(:pos))
        @test first(pl_pos.x) == Bijectors.Log{0}()(l)
        @test value(pl_pos) == Bijectors.Exp{0}()(Bijectors.Log{0}()(l))
        pv_pos = Parameter(v, Val(:pos))
        @test pv_pos.x == Bijectors.Log{1}()(v)
        @test value(pv_pos) == Bijectors.Exp{1}()(Bijectors.Log{1}()(v))
        pW_pos = Parameter(W, Val(:pos))
        @test pW_pos.x == Bijectors.Log{2}()(W)
        @test value(pW_pos) == Bijectors.Exp{2}()(Bijectors.Log{2}()(W))
    
        # check input domain
        @test_throws DomainError Parameter(-3.0, Val(:pos))
        @test_throws DomainError Parameter(9.0, inv(Bijectors.Logit(1.3, 6.0)))
    end
end



