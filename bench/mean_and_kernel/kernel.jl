@benchset "kernel" begin

    @benchset "ZeroKernel{Float64}()" begin
        create_benchmarks(ZeroKernel{Float64}(); grads=false)
    end
    # @benchset "OneKernel()" begin
    #     create_benchmarks(OneKernel(); grads=false)
    # end
    # @benchset "EQ" begin
    #     @benchset "Real CPU" create_benchmarks(EQ())
        # @benchset "Real GPU" create_benchmarks(
        #     EQ();
        #     x=randn(Float32), x′=randn(Float32),
        #     x̄s=[CuArray{Float32}(randn(N)) for N in Ns()],
        #     x̄′s=[CuArray{Float32}(randn(N)) for N in Ns()],
        # )

    #     for D in Ds()
    #         @benchset "ColsAreObs (D=$D) CPU" create_benchmarks(
    #             EQ();
    #             x=randn(D), x′=randn(D),
    #             x̄s=[ColsAreObs(randn(D, N)) for N in Ns()],
    #             x̄′s=[ColsAreObs(randn(D, N)) for N in Ns()],
    #         )
    #     end

    #     # See https://github.com/FluxML/Zygote.jl/issues/44
    #     @benchset "EQ Almost-Toeplitz" create_benchmarks(
    #         EQ();
    #         x=randn(), x′=randn(),
    #         x̄s=[range(-randn(), step=randn(), length=N) for N in Ns()],
    #         x̄′s=[range(-randn(), step=randn(), length=N) for N in Ns()],
    #     )

    #     δ = randn()
    #     @benchset "EQ Toeplitz" create_benchmarks(
    #         EQ();
    #         x=randn(), x′=randn(),
    #         x̄s=[range(-randn(), step=δ, length=N) for N in Ns()],
    #         x̄′s=[range(-randn(), step=δ, length=N) for N in Ns()],
    #     )
    # end

    # @benchset "PerEQ" begin
    #     @benchset "Real CPU" create_benchmarks(PerEQ())
    #     # @benchset "Real GPU" create_benchmarks(PerEQ();
    #     #     x = randn(Float32), x′=randn(Float32),
    #     #     x̄s=[CuArray{Float32}(randn(N)) for N in Ns()],
    #     #     x̄′s=[CuArray{Float32}(randn(N)) for N in Ns()],
    #     # )
    # end
end
