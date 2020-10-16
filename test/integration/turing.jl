# Ensure that using Stheno inside of Turing doesn't error.
@testset "turing" begin

    # make some fake data
    l = 0.4
    σ² = 1.3
    σ²_n = 0.05 # noise
    x_ = collect(range(-4.0, 4.0; length=10))
    y_ = rand(GP(σ² * stretch(Matern52(), 1 / l), GPC())(x_, σ²_n))

    # prior model
    @model gp0(y,x) = begin
        σ² ~ LogNormal(0, 1)
        l ~ LogNormal(0, 1)
        σ²_n ~ LogNormal(0, 1)  
        k =  σ² * stretch(Matern52(), 1 ./ l)
        f = GP(k, GPC())
        y ~ f(x, σ²_n + 1e-3)
    end

    # sample from posterior
    m = gp0(y_, x_)
    chain = sample(m, HMC(0.01, 100), 2)
end
