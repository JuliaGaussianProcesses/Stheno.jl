@testset "additive_gp" begin
    @testset "arbitrary indices" begin
        f = @gppp let
            f1 = GP(SEKernel())
            f2 = GP(SEKernel())
            f3 = additive_gp((f1, f2), [2:3, 1])
        end
        x_raw = ColVecs(randn(3, 7))
        x1_raw = ColVecs(x_raw.X[2:3, :])
        x2_raw = x_raw.X[1, :]
        x = vcat(
            GPPPInput(:f1, x1_raw),
            GPPPInput(:f2, x2_raw),
            GPPPInput(:f3, x_raw),
        )
        y = rand(f(x, 1e-9))

        @test y[1:7] + y[8:14] ≈ y[15:21] rtol=1e-3

        z = GPPPInput(:f3, ColVecs(3 * randn(3, 4)))
        test_internal_abstractgps_interface(MersenneTwister(123456), f, x, z; jitter=1e-15)
    end
    @testset "regular indices" begin
        f = @gppp let
            f1 = GP(SEKernel())
            f2 = GP(Matern52Kernel())
            f3 = additive_gp((f1, f2))
        end
        x_raw = ColVecs(randn(2, 7))
        x1_raw = x_raw.X[1, :]
        x2_raw = x_raw.X[2, :]
        x = vcat(
            GPPPInput(:f1, x1_raw),
            GPPPInput(:f2, x2_raw),
            GPPPInput(:f3, x_raw),
        )
        y = rand(f(x, 1e-9))
        @test y[1:7] + y[8:14] ≈ y[15:21] rtol=1e-3

        z = GPPPInput(:f3, ColVecs(3 * randn(2, 4)))
        test_internal_abstractgps_interface(MersenneTwister(123456), f, x, z; jitter=1e-15)
    end
end
