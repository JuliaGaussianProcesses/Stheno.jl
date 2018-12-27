@benchset "mean" begin
    @benchset "ZeroMean{Float64}()" create_benchmarks(ZeroMean{Float64}(); grads=false)
    @benchset "OneMean()" create_benchmarks(OneMean(); grads=false)
    @benchset "CustomMean(sin)" create_benchmarks(CustomMean(sin))
    @benchset "CustomMean(cos)" create_benchmarks(CustomMean(cos))
    @benchset "sin + cos" create_benchmarks(CustomMean(sin) + CustomMean(cos))
    @benchset "5 * CustomMean(sin)" create_benchmarks(5.0 * CustomMean(sin))
end
