# This doesn't currently work.
# https://github.com/JuliaGaussianProcesses/Stheno.jl/issues/222

using AbstractGPs
using Plots
using Random
using RDatasets
using Stheno
using Turing

rng = MersenneTwister(5)
T = Float64


# Use RDatasets to load a binary classification dataset, we use features from this dataset to predict it's color
# The crabs data frame has 200 rows and 8 columns, describing 5 morphological measurements on 50 crabs 
# each of two colour forms and both sexes, of the species Leptograpsus variegatus collected at Fremantle, W. Australia.
crabs = dataset("MASS","crabs")
crabs = crabs[shuffle(1:size(crabs, 1)), :]
train = crabs[1:div(end,2), :]
test = crabs[div(end,2)+1:end, :]

train_y = Array{Bool}(undef,size(train, 1))
train_y[train.Sp.=="B"].=0
train_y[train.Sp.=="O"].=1
train_x = Matrix(transpose(convert(Array,train[:,4:end])))

test_y = Array{Bool}(undef, size(test, 1))
test_y[test.Sp.=="B"].=0
test_y[test.Sp.=="O"].=1
test_x = Matrix(transpose(convert(Array, test[:, 4:end])))



# The probabilistic model for the binary classification problem can be formulated as:
# f ~ GP(μ, K)
# y|f ~ Bernoulli(sigmoid(f))
# This model is build using Turing.jl, please refer to https://turing.ml/dev/docs/using-turing/get-started for details.

σ(x) = one(T) / (one(T)+exp(-x))

function build_gp(logl, σ², X)
    ard_eq_kernel = σ² * stretch(SEKernel(), exp.(-logl))
    gp = GP(ard_eq_kernel, GPC())
    prior = gp(ColVecs(X), T(0.01))
    gp, prior
end

# The Turing model used to estimate the posterior distribution,
# the latent variable is f & the parameter is logl
@model gpc_learn(X, y) = begin
    logl ~ Normal(T(0.0), T(2.0))
    _, prior = build_gp(logl, T(1.0), X)
    f ~ prior
    for i in eachindex(y)
      	y[i] ~ Bernoulli(σ(f[i]))
    end
end

# Function used to infer the label for newly inputs
# NOTE: for simplicity, we use MAP estimation here instead of integrate over the latent and parameters
function gpc_infer(x, logl, Xtrain, fsamples)
    nsamples = size(fsamples, 2)
    fxs = []
    for i in 1:nsamples
        gp, prior = build_gp(logl[i], T(1.0), Xtrain)
        conditioned_gp = gp | Obs(prior, fsamples[:, i])
        posterior = conditioned_gp(ColVecs(x))
        push!(fxs, mean.(marginals(posterior)))
    end
    fx_mean = vec(mean(hcat(fxs...), dims=2))
    p = σ.(fx_mean)
    y = Int.(p .> T(0.5))
    y
end

# Marginalize over non Gaussian likelihood is intractable, 
# we use MCMC to compute the approximate posterior 
model = gpc_learn(train_x, train_y)
mcmc_samples = sample(model, HMC(0.01, 10), 5000);

# store the sampled parameter & latent variable, descard the first 1000 samples
logl_df = mcmc_samples[:logl]
logl = vec(logl_df.value.data)
logl = Array{T}(logl)
reserve_logl = logl[1001:end]

fsamples_df = mcmc_samples[:f]
fsamples = Matrix(transpose(dropdims(fsamples_df.value.data, dims=3)))
fsamples = convert.(T, fsamples)
reserve_fsamples = fsamples[:, 1001:end]


# prediction
pred_y = gpc_infer(test_x, reserve_logl, train_x, reserve_fsamples)
# determine the accuracy of our prediction
function accuracy(pred_y, y)
    N = length(y)
    N_neq = sum(abs.(pred_y .- y))
    r = T(1.0) - N_neq / N
    r
end

@show accuracy(pred_y, test_y)
