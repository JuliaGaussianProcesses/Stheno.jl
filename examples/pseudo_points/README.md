# Approximate inference with Pseudo Points

Also known as sparse approximations, inducing-point approimations, and probably a few more names.

Here we demonstrate how to perform efficient approximate inference in large data sets when a small number of pseudo-observations can be utilised to represent the posterior.

Titsias, 2009, is the de-facto standard approach to this type of approximation, and is the algorithm utilised here. There are, however, plenty of other pseudo-point approximations exist in the literature, and wouldn't be particularly difficult to add. If you would like to see them implemented, please either do so and open a PR, or raise an issue on the matter!
