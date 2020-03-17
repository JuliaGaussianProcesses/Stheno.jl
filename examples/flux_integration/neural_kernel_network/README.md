# Neural kernel network
When applying a neural network to data, we compute different linear features of data and pass the result to a nonlinear
activation function. A similar thing can be done with kernels as well. In particular, kernels can be combined according to the following rules ( $k_1,\,k_2$ are kernel functions ):

1. For $\lambda_1,\,\lambda_2\in\mathbb{R}^+$, $\lambda_1 k_1+\lambda_2 k_2$ is also a valid kernel function

2. The product $k_1k_2$ is a kernel function

Therefore, we can introduce a neural network like structure, which consists superposition, product and nonlinear activation operations ( similar to the idea fo sum and product network ). Acting such a network on top of some basic kernels, e.g. exponential quardratic kernel, periodic kernel, linear kernel, etc., will results in a new kernel function which is able to extract more complicated features. In addition, the neural kernel network is differentiable w.r.t it's parameters, which enables us to determine it via gradient based optimization methods. This also illuminates previous black-box kernel formation process.

## Reference
---
[1] Shengyang Sun, Guodong Zhang, Chaoqi Wang, Wenyuan Zeng, Jiaman Li, Roger Grosse (2018) [Differentiable Compositional Kernel Learning for Gaussian Processes](https://arxiv.org/pdf/1806.04326.pdf)

[2] Duvenaud, D., Lloyd, J. R., Grosse, R., Tenenbaum, J. B., and Ghahramani, Z. (2013) [Structure discovery in nonparametric regression through compositional kernel search](https://arxiv.org/abs/1302.4922)
