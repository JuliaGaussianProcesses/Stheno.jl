

using LinearAlgebra, Stheno, Flux, Zygote, DelimitedFiles, Statistics
using Plots; pyplot();
using Random; Random.seed!(4);

######################################################
# Data loading 
## read AirPass data
data = readdlm("AirPassengers.csv", ',')
year = data[2:end,2]; passengers = data[2:end,3];
## Split the data into training and testing data
oxtrain = year[year.<1958]; oytrain = passengers[year.<1958];
oxtest = year[year.>=1958]; oytest = passengers[year.>=1958];

##data preprocessing
### standardize X and y
xtrain_mean = mean(oxtrain)
ytrain_mean = mean(oytrain)
xtrain_std = std(oxtrain)
ytrain_std = std(oytrain)
xtrain = @. (oxtrain-xtrain_mean)/xtrain_std
ytrain = @. (oytrain-ytrain_mean)/ytrain_std

xtest = @. (oxtest-xtrain_mean)/xtrain_std
ytest = @. (oytest-ytrain_mean)/ytrain_std

## input data
Xtrain = reshape(xtrain, 1, length(xtrain))
Xtest = reshape(xtest, 1, length(xtest))
Year = hcat(Xtrain, Xtest)
Passengers = vcat(ytrain, ytest)
######################################################

plt = plot(xlabel="Year", ylabel="Airline Passenger number", legend=true)
scatter!(plt, oxtrain, oytrain, label="Observations(train)", color=:black)




######################################################
# Build kernel with Neural Kernel Network
## kernel length scale initialization
function median_distance_local(x)
    n = length(x)
    dist = []
    for i in 1:n
        for j in i:n
            push!(dist, abs(x[j]-x[i]))
        end
    end
    median(dist)
end
l = median_distance_local(xtrain)

## kernel parameter constraint
g1(x) = exp(-x)
g2(x) = exp(x)

## define kernels
iso_lin_kernel1 = stretch(Linear(), log(1.0), g1)
iso_per_kernel1 = scale(stretch(PerEQ(log(l), g2), log(l), g1), log(1.0), g2)
iso_eq_kernel1 = scale(stretch(EQ(), log(l/4.0), g1), log(1.0), g2)
iso_rq_kernel1 = scale(stretch(RQ(log(0.2), g2), log(2.0*l), g1), log(1.0), g2)
iso_lin_kernel2 = stretch(Linear(), log(1.0), g1)
iso_rq_kernel2 = scale(stretch(RQ(log(0.1), g2), log(l), g1), log(1.0), g2)
iso_eq_kernel2 = scale(stretch(EQ(), log(l), g1), log(1.0), g2)
iso_per_kernel2 = scale(stretch(PerEQ(log(l/4.0), g2), log(l/4.0), g1), log(1.0), g2)


# define network
linear1 = LinearLayer(8, 8)
prod1 = ProductLayer(2)
linear2 = LinearLayer(4, 4)
prod2 = ProductLayer(2)
linear3 = LinearLayer(2, 1)

## NKN
player = Primitive(iso_lin_kernel1, iso_per_kernel1, iso_eq_kernel1, iso_rq_kernel1,
                   iso_lin_kernel2, iso_rq_kernel2, iso_eq_kernel2, iso_per_kernel2)
nn = chain(linear1, prod1, linear2, prod2, linear3)
nkn = NeuralKernelNetwork(player, nn)
#############################################################


# Do some common calculation
σ²_n = 0.1                                           # specify Gaussian noise
gp = GP(nkn, GPC())                                  # define GP
loss(m, x, y) = -logpdf(m(ColVecs(x), σ²_n), y)      # define loss & compute negative log likelihood
loss(gp, Xtrain, ytrain)
∂gp, = gradient(m->loss(m, Xtrain, ytrain), gp)      # compute derivative of loss w.r.t GP parameters

# extract all parameters from the GP model
l_ps = parameters(gp) |> length
# extract the corresponding gradients from the derivative ( or conjugate of GP model )
l_∂ps = extract_gradient(gp, ∂gp) |> length
# make sure parameters and gradients are in one-to-one correspondence
@assert l_ps == l_∂ps


#############################################################
# Optimize GP parameters w.r.t training data
using Flux.Optimise: update!

optimizer = ADAM(0.001)
L = []
for i in 1:5000
    nll = loss(gp, Xtrain, ytrain)
    push!(L, nll)
    if i==1 || i%200 == 0
        @info "step=$i, loss=$nll"
    end
    ps = parameters(gp)
    ∂gp, = gradient(m->loss(m, Xtrain, ytrain), gp)
    
    Δps = extract_gradient(gp, ∂gp)
    update!(optimizer, ps, Δps)
    dispatch!(gp, ps)                                # dispatch! will update the GP model with updated parameters
end

# you can view the loss curve
# plot(L, legend=false)
#############################################################


#############################################################
# make prediction
function predict(gp, X, Xtrain, ytrain)
    gp_Xtrain = gp(ColVecs(Xtrain), σ²_n)
    posterior = gp | Obs(gp_Xtrain, ytrain)
    posterior(ColVecs(X))
end

posterior = predict(gp, Year, Xtrain, ytrain)
post_dist = marginals(posterior)
pred_y = mean.(post_dist)
var_y = std.(post_dist)

pred_oy = @. pred_y*ytrain_std+ytrain_mean
pred_oσ = @. var_y*ytrain_std

plot!(plt, year, pred_oy, ribbons=3*pred_oσ, title="Time series prediction",label="95% predictive confidence region")
scatter!(plt, oxtest, oytest, label="Observations(test)", color=:red)
display(plt)
##############################################################








