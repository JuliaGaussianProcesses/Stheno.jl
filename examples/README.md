# Examples Overview

There are numerous examples of things that can be done with Stheno.jl in this directory. The `getting_started` sub-directory, you should have the pre-requisites to understand the other folders, each of while build on `getting_started` in 

Note that each sub-directory contains its own environment that specifies a particular version of Stheno and other dependencies. As such, each example _should_ certainly be runnable. It might not, however, be runnable on the most recent version of Stheno. If you encounter this, it likely means that someone forgot to check that all of the examples still run when a breaking change was made. This should be considered a bug and an issue raised, or PR open to fix the problem!

Below we provide a brief description of each of the sub-directories.

- `getting_started`: the most fundamental Stheno.jl functionality. If you're not comfortable with the content of this folder, you likely won't be with the rest of them.
- `pseudo_points`: covers inducing-point / sparse / pseudo-point approximations.
- `basic_gppp`: basic toy examples of the functionality that we call Gaussian process Probabilistic Programming (GPPP).
- `gppp_and_pseudo_points`: combine GPPP and pseudo-point approximations to do interesting things. This is a WIP -- it doesn't work properly yet.
- `flux_integration`: indicate how Stheno.jl work with Flux.jl. Currently we have an example shows how to use Flux's neural network as a feature extractor and then use GP to do regression, Stheno's power is not limited to this, we are working on adding more functionalities and more examples.
- `turing_integration`: a binary classification example shows how Stheno.jl, with the help of Turing.jl, can be used in non-Gaussian likelihood problems.
