@testset "algebra" begin

let
    # Zero-means return more zero-means.
    c = ConstantMean(5.0)
    @test f + f === f
    @test f + c == c
    @test c + f == c
    @test f * f === f
    @test f * c === f
    @test c * f === f
end

end
