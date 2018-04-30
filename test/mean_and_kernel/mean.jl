@testset "mean" begin

    # Test base kernels.
    let
        @test CustomMean(sin) == CustomMean(sin)
        @test CustomMean(cos) == CustomMean(cos)
        @test CustomMean(sin) != CustomMean(cos)
        @test ZeroMean{Float64}() == ZeroMean{Int}()
        @test ConstantMean(1.0) == ConstantMean(1.0)
    end

    # # Test that composition works as expected.
    # let
    #     @test (ZeroMean() + ZeroMean())(1.0) == 0.0
    #     @test (ZeroMean() + ZeroMean()) == (ZeroMean() + ZeroMean())
    #     @test (OneMean() + ConstantMean(2.5)) == OneMean() + 2.5
    #     @test (OneMean() + ConstantMean(1.3))(6.451) ≈ 2.3
    #     @test (ConstantMean(2.3) + OneMean()) == 2.3 + OneMean()

    #     @test (ZeroMean() * OneMean())(5.34625) == 0.0
    #     @test (OneMean() * 5)(:foo) == 5
    #     @test OneMean() * 4 == OneMean() * ConstantMean(4)
    #     @test (4 * OneMean()) == ConstantMean(4) * OneMean()
    # end

    # # Test FiniteMean.
    # let
    #     @test FiniteMean(CustomMean(sin), [5.0, 6.0, 7.0])(2) == sin(6.0)
    #     @test FiniteMean(CustomMean(cos), [1.0, 2.0, 3.0]).([1, 2, 3]) == cos.([1.0, 2.0, 3.0])
    # end
end
