# Moment Unfolding

Consider `S_T, S_M, D_T, D_M` where `T` stands for truth, `R` stands for measured, `S` stands for simulation and `D` stands for data.  `S_T, D_T\in\mathbb{R}^N` and `S_M, D_M\in\mathbb{R}^M`.  We observe `D_M` and want to use `(S_T,S_M)` to infer `D_T`.  This is the usual unfolding problem.  Suppose we only want to know moments of `D_T`?  For instance, let's say we want to know `E[D_T[0] | D_T[1]]` where `D_T[0]` and `D_T[1]` are the first and second elements of `D_T`.  One could solve this by doing the full unfolding of `D_T` and then computing momemts, but this is overkill - the question is if we can design a dedicated approach for moment unfolding.

An elegent solution to this is inspired by the Boltzman distribution.  That distribution is constructed as the minimal entropy distribution with a given mean.  We can instead construct a distribution which has the smallest KL-divergence with the simulation truth, but has a given set of moments that match the data truth.  Working through the functional optimization, one can show that this can be achieved by training a GAN-like model where the "generator" is simply a reweighter function that has the specific form `exp(\lambda_1 x + \lambda_2 x^2 + ...)` where the `\lambdas` correspond to which moments you want to match.  

This is illustrated using examples from Z+jets in the notebook in this repo.  

However, why stop there?  Can we use this approach to unfold the full phase space in one go without having to do the iterative approach in OmniFold?  A key challenge is that there is a freedom to pick the partition function, which does not change the analytic results, but seems to have a big effect practically.  Can we understand this?  The last examples in the notebook show the setup.
