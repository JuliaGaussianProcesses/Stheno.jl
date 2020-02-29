# Set up the environment to run this example. Make sure you're within the folder that this
# file lives in.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


using Stheno
using Plots; pyplot();
using Random; Random.seed!(4);
using Flux
using Zygote
using DelimitedFiles
using Statistics


# read AirPass data
data = readdlm("AirPassengers.csv", ',')
year = data[2:end,2]; passengers = data[2:end,3];
# Split the data into training and testing data
oxtrain = year[year.<1958]; oytrain = passengers[year.<1958];
oxtest = year[year.>=1958]; oytest = passengers[year.>=1958];

#data preprocessing
## standardize X and y
xtrain_mean = mean(oxtrain)
ytrain_mean = mean(oytrain)
xtrain_std = std(oxtrain)
ytrain_std = std(oytrain)
xtrain = @. (oxtrain-xtrain_mean)/xtrain_std
ytrain = @. (oytrain-ytrain_mean)/ytrain_std

xtest = @. (oxtest-xtrain_mean)/xtrain_std
ytest = @. (oytest-ytrain_mean)/ytrain_std

## input data
Xtrain = reshape(xtrain, 1, length(xtrain));
Xtest = reshape(xtest, 1, length(xtest));
Year = hcat(Xtrain, Xtest);
Passengers = vcat(ytrain, ytest)


# kernel parameter initialization
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



# construct kernels
iso_lin_kernel1 = stretch(Linear(), [0.0])
iso_per_kernel1 = [log(1.0)] * stretch(PerEQ([log(l)]), [log(l)])
iso_eq_kernel1 = [log(1.0)] * stretch(EQ(), [log(l/4.0)])
iso_rq_kernel1 = [log(1.0)] * stretch(RQ([log(0.2)]), [log(2.0*l)])
iso_lin_kernel2 = stretch(Linear(), [0.0])
iso_rq_kernel2 = [log(1.0)] * stretch(RQ([log(0.1)]), [log(l)])
iso_eq_kernel2 = [log(1.0)] * stretch(EQ(), [log(l)])
iso_per_kernel2 = [log(1.0)] * stretch(PerEQ([log(l/4.0)]), [log(l/4.0)])


# sum product network
linear1 = LinearLayer(8, 8)
linear2 = LinearLayer(4, 4)
linear3 = LinearLayer(2, 1)

# NKN
player = Primitive(iso_lin_kernel1, iso_per_kernel1, iso_eq_kernel1, iso_rq_kernel1,
                     iso_lin_kernel2, iso_rq_kernel2, iso_eq_kernel2, iso_per_kernel2)
nn = Chain(linear1, Product, linear2, Product, linear3)
nkn = NeuralKernelNetwork(player, nn)


# build GP model
σ²_n = 0.1
gp = GP(nkn, GPC())
gp_Xtrain = gp(ColVecs(Xtrain), σ²_n)
ps = params(nkn)

# optimize
using Flux.Optimise: update!

optimizer = ADAM(0.001)
loss = []
for i in 1:5000
    ll = .-logpdf(gp_Xtrain, ytrain)
    push!(loss, ll)
    if i==1 || i%100 == 0
        @info "step=$i, loss=$ll"
    end
    gs = gradient(()->.-logpdf(gp_Xtrain, ytrain), ps)
    for p in ps
        update!(optimizer, p, gs[p])
    end
end

display(plot(loss))


# predict
function predict(X, Xtrain, ytrain)
    noisy_prior = gp(ColVecs(Xtrain), σ²_n)
    posterior = gp | Obs(noisy_prior, ytrain)
    posterior(ColVecs(X))
end

posterior = predict(Year, Xtrain, ytrain)
pred_y = mean(posterior)
pred_oy = @. pred_y*ytrain_std+ytrain_mean

plt = plot(xlabel="Year", ylabel="Airline Passenger number", legend=true)
plot!(plt, year, pred_oy, title="Time series prediction",label="95% predictive confidence region")
scatter!(plt, oxtest, oytest, label="Observations(test)", color=:red)
scatter!(plt, oxtrain, oytrain, label="Observations(train)", color=:black)
display(plt)

