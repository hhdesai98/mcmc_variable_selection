# mcmc_variable_selection

This repo contains an MCMC variable selection algorithm for logistic regression. The dataset contains 60 covariates and a response variable y. The parallelR22.cpp file contains the algorithm itself, which will select randomly variables and rank them according to their Marginal Likelihood (calculated through monte carlo integration).  
