# mcmc_variable_selection

This repo contains an MCMC variable selection algorithm for univariate logistic regression. The dataset contains 60 covariates and a response variable y. The parallelR22.cpp file contains the algorithm itself, which will select variables and rank models according to their Marginal Likelihood (calculated through monte carlo integration and laplace approximations).  
