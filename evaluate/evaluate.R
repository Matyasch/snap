# This file contains helper functions used by evaluate.py

library(pcalg)


true_binary_effect <- function(treatment, outcome, amat, cpt,
                               n = 10000, seed = 0) {
    #' Compute binary ATE
    #'
    #' Compute the true ATE of a binary effect empriically with interventions
    #'
    #' @param treatment The index of the treatment variable.
    #' @param outcome The index of the outcome variable.
    #' @param amat The adjacency matrix of the graph.
    #' @param cpt The conditional probability table of the graph.
    #' @param n The number of samples to generate for empirical estimation.
    #' @param seed The random seed to use for sample generation.
    #' @return The true ATE of the binary effect.

    set.seed(seed)
    # Create mutilated graph
    mutilated <- amat
    mutilated[, treatment] <- 0 # remove parents
    bng <- bnlearn::as.bn(as(mutilated != 0, "graphNEL"))

    # Generate interventional data for do(treatment = 0)
    cpt_0 <- cpt
    cpt_0[[treatment]] <- matrix(c(1, 0), ncol = 2)
    bnfit_0 <- bnlearn::custom.fit(bng, dist = cpt_0)
    data_0 <- bnlearn::rbn(bnfit_0, n = n)
    data_0 <- match(as.matrix(data_0), LETTERS) - 1
    data_0 <- matrix(data_0, ncol = ncol(amat))
    # E[outcome | do(treatment = 0)]
    mean_0 <- mean(data_0[, as.numeric(outcome)])

    # Generate interventional data for do(treatment = 1)
    cpt_1 <- cpt
    cpt_1[[treatment]] <- matrix(c(0, 1), ncol = 2)
    bnfit_1 <- bnlearn::custom.fit(bng, dist = cpt_1)
    data_1 <- bnlearn::rbn(bnfit_1, n = n)
    data_1 <- match(as.matrix(data_1), LETTERS) - 1
    data_1 <- matrix(data_1, ncol = ncol(amat))
    # E[outcome | do(treatment = 1)]
    mean_1 <- mean(data_1[, as.numeric(outcome)])

    # ATE = E[outcome | do(treatment = 1)] - E[outcome | do(treatment = 0)]
    effect <- mean_1 - mean_0
    return(effect)
}


get_adjustmet_set <- function(treatment, outcome, amat) {
    #' Get adjustment set
    #'
    #' Get adjustment set for treatment and outcome..
    #' The function first tries to obtain the optimal adjustment set..
    #' If that fails, it falls back to the canonical adjustment set..
    #' @param treatment The index of the treatment variable.
    #' @param outcome The index of the outcome variable.
    #' @param amat The adjacency matrix of the graph.
    #' @return The adjustment set for treatment and outcome.

    tryCatch(
        {
            return(pcalg::optAdjSet(as(amat, "graphNEL"), treatment, outcome))
        },
        error = function(e) {
            adj_set <- pcalg::adjustment(
                t(amat), "cpdag", treatment, outcome, "canonical"
            )
            if (length(adj_set) == 0) {
                return(NULL)
            } else {
                return(unlist(adj_set))
            }
        }
    )
}


shd <- function(oracle, estimate) {
    #' Structural Hamming Distance
    #'
    #' Compute the Structural Hamming Distance between two graphs
    #' @param oracle Adjacency matrix of the true graph.
    #' @param estimate Adjacency matrix of the estimated graph.
    #' @return The Structural Hamming Distance between the two graphs.

    return(pcalg::shd(as(oracle, "graphNEL"), as(estimate, "graphNEL")))
}
