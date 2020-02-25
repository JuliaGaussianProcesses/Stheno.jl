# Set up the environment to run this example. Make sure you're within the folder that this
# file lives in.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


using Plots, Random, Stheno, Flux, Zygote
rng = MersenneTwister(5)
T = Float64


# define the step function 
# step_func(x) = 0.0 if x<=0, 1.0 if x>0
step_func(x) = (ϵ=T(0.01)*randn(rng, T); x>T(0.0) ? T(1.0)+ϵ : T(0.0)+ϵ)

# prepare data
## training data, drawn from Normal(0, 1)
train_X = randn(rng, T, 100)
## Flux requires data to be at least two dimension
Xtrain = reshape(train_X, 1, :)
## evaluate to get the function value
train_y = step_func.(train_X)

## test data drawn from Uniform(-5, 5)
test_X = Array(-5.0:0.02:5.0)
Xtest = reshape(test_X, 1, :)
test_y = step_func.(test_X)



# Build a MLP that perform domain transformation to input data
# for Flux usage, please refer to: https://fluxml.ai/Flux.jl/stable/
mlp = Chain(Dense(1, 6, relu), Dense(6, 2, relu)) |> (T==Float32 ? f32 : f64)
# extract MLP parameters
θ_mlp = params(mlp)

# Build a GP model with Stheno
# here we consider using anisotropic EQ kernel, this kernel contains to hyperparameters: length scale l & scaling factor γ
# these hyperparameters are positive, we add on this restriction by resetting them in log-scale
# for Stheno usage, please refer to: https://github.com/willtebbutt/Stheno.jl/tree/wct/example-revamp/examples/getting_started
logl = randn(rng, T, 2)
logγ = T[0.0]

function build_gp(logl, logγ)
	ard_se_kernel = exp(T(2.0)*logγ[1])*stretch(EQ(), exp.(-logl))
	gp = GP(T(0.0), ard_se_kernel, GPC())
  gp
end

# Since we always assume our data to be noisy, we model this noise by λ, also in log-scale
logλ = T[0.01]

# Collect MLP and GP parameters
ps = Params([logl, logγ, logλ, θ_mlp...])



# When training, we always specifies a loss function to optimize, for GP we use negative log-likelihood
# !!! NOTE:
# Stheno and Flux have slightly different conventions regarding their data. In particular
# - Flux assumes that you'll provide a `Matrix` of data, in which each column is an
#   observation.
# - Stheno assumes that you'll provide an `AbstractVector`, where each element is an
#   observation. To handle multi-dimensional inputs we have the `ColVecs` object, that
#   literally just wraps a `Matrix` and tells Stheno to pretend is a vector of vectors. This
#   is helpful to remove some ambiguities that arrise if you don't do this.
function NLL(X, y)
  Z = mlp(X)
  gp = build_gp(logl, logγ)
	margin_lik = gp(ColVecs(Z), exp(T(2.0)*logλ[1]))
  nll = -logpdf(margin_lik, y)
  nll
end



# Training the overall model with Flux optimizers
using Flux.Optimise: update!

train_data = (Xtrain, train_y)
opt = ADAGrad()
nlls = []
for i in 1:1500
  nll = NLL(train_data...)
  push!(nlls, nll)
  gs = gradient(()->NLL(train_data...), ps)
  for p in ps
    update!(opt, p, gs[p])
  end
end

loss_plot = plot(xlabel="Epoches", ylabel="Negative log-likelihood", legend=false)
plot!(loss_plot, nlls)
png(loss_plot, "loss.png")



# Visualize the performance of our model
function predict(X, Xtrain, ytrain)
    Z = mlp(X); Ztrain = mlp(Xtrain)
    gp = build_gp(logl, logγ)
		noisy_prior = gp(ColVecs(Ztrain), exp(T(2.0)*logλ[1]))
    posterior = gp | Obs(noisy_prior, ytrain)
    posterior(ColVecs(Z))
end

posterior = predict(Xtest, Xtrain, train_y)
post_dist = marginals(posterior)
pred_y = mean.(post_dist)
var_y = std.(post_dist)


predict_plot = plot(legend=true, xlabel="x", ylabel="y", ylim=(-0.5, 1.5))
plot!(predict_plot, test_X, pred_y, ribbons=3*var_y, st=:line, fillalpha=0.3, color=:blue, label="NNGP")
plot!(predict_plot, train_X, train_y, st=:scatter, color=:red, lw=5, label="Training set")
png(predict_plot, "predict.png")
