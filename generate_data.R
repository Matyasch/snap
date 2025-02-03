generate_degseq <- function(nodes, exp_degree, max_degree) {
    #' Generate a degree sequence
    #'
    #' Generate a degree sequence with a given expected and maximum degree.
    #' @param nodes The number of nodes in the graph.
    #' @param exp_degree The expected degree of each node.
    #' @param max_degree The maximum degree of each node.
    #' @return A degree sequence.

    p <- exp_degree / nodes
    seq <- sapply(1:nodes, function(i) {
        deg <- sum(rbinom(nodes, 1, p))
        while (deg < 1 || deg > max_degree) {
            deg <- sum(rbinom(nodes, 1, p))
        }
        return(deg)
    })
    return(seq)
}

generate_dag <- function(nodes, exp_degree, max_degree, connected = TRUE) {
    #' Generate a directed acyclic graph
    #'
    #' Generate a directed acyclic graph with a given number of nodes,
    #' expected and maximum degree.
    #' @param nodes The number of nodes in the graph.
    #' @param exp_degree The expected degree of each node.
    #' @param max_degree The maximum degree of each node.
    #' @param connected Whether the graph should be connected.
    #' @return A directed acyclic graph in igraph.

    method <- ifelse(connected, "vl", "simple.no.multiple")
    g <- NULL
    while (is.null(g)) {
        seq <- generate_degseq(nodes, exp_degree, max_degree)
        if (igraph::is_graphical(seq)) {
            try( # Try to generate a DAG with the sampled degree sequence
                {
                    g <- igraph::sample_degseq(seq, method = method)
                },
                silent = TRUE
            )
        }
    }
    g <- igraph::as.directed(g, mode = "acyclic")
    return(g)
}

get_definite_reach <- function(amat) {
    #' Definite reachibility
    #'
    #' Compute the definite reachibility matrix of a CPDAG.
    #' @param amat The adjacency matrix of the CPDAG.
    #' @return The definite reachibility matrix.

    amat[amat == t(amat)] <- 0 # remove undirected edges
    reach <- expm::expm(amat) > 0
    diag(reach) <- FALSE
    return(reach | t(reach))
}

get_identifiable_pairs <- function(amat, min_adj_size) {
    #' Identifiable pairs
    #'
    #' Compute the identifiable pairs of nodes of a CPDAG.
    #' @param amat The adjacency matrix of the CPDAG.
    #' @param min_adj_size The minimum size of the adjustment set required
    #' to considered.
    #' @return The matrix of identifiable pairs.

    idable <- t(expm::expm(amat) == 0) # unreachable
    labels <- colnames(amat)
    g <- pcalg::pcalg2dagitty(amat, labels)
    ind <- which(!idable, arr.ind = TRUE)
    for (i in seq_len(nrow(ind))) {
        y <- ind[i, 2]
        x <- ind[i, 1]
        a <- dagitty::adjustmentSets(
            g, labels[x], labels[y], "minimal", "total",
            max.results = 1
        )
        idable[x, y] <- length(a) > 0 && length(a[[1]]) >= min_adj_size
    }
    return(idable & t(idable))
}

sample_clique <- function(amat, n) {
    #' Sample a clique
    #'
    #' Sample a clique of size n from the graph defined by the adjacency matrix.
    #' We sample cluqies randomly because enumerating all of them with
    #' (igraph::cliques) is too slow for large graphs.
    #' @param amat The adjacency matrix of the graph.
    #' @param n The size of the clique.
    #' @return The sampled clique.

    rec_build <- function(amat, n, poss, cands) {
        # Check if done
        if (length(cands) == n) {
            return(cands)
        }
        # Get common neighbors of all current candidates
        if (length(cands) == 1) {
            common <- which(amat[cands, ] == 1)
        } else {
            common <- which(apply(amat[cands, ], 2, prod) == 1)
        }
        # Keep only those with more than n neighbors
        common <- intersect(common, poss)
        if (length(common) == 0) { # If no such neighbors, return NULL
            return(NULL)
        } else if (length(common) == 1) { # If only one neighbor, add it
            return(rec_build(amat, n, poss, c(cands, common)))
        } else { # If multiple neighbors, loop through them
            for (cand in sample(common)) {
                return(rec_build(amat, n, poss, c(cands, cand)))
            }
        }
    }

    poss <- which(rowSums(amat) >= n)
    if (length(poss) < n) {
        return(NULL)
    }
    for (nb in sample(poss, min(length(poss), 1000))) {
        cands <- rec_build(amat, n, poss, nb)
        if (!is.null(cands)) {
            return(cands)
        }
    }
    return(NULL)
}

get_identifiable_targets <- function(dag, targets, min_adj_size) {
    #' Identifiable targets
    #'
    #' Get a set of identifiable targets given a DAG.
    #' @param dag The causal DAG of class graph.
    #' @param targets The number of targets to sample.
    #' @param min_adj_size The minimum size of the adjustment set required
    #' to be considered identifiable.

    amat <- t(as(pcalg::dag2cpdag(dag), "matrix"))
    idable <- get_identifiable_pairs(amat, min_adj_size)
    reach <- get_definite_reach(amat)
    for (i in 0:1000) { # Try 1000 times then fail
        # All effects (including no effect) should be identifiable
        clique <- sample_clique(idable, targets)
        # All targets should be ancestors/descedants of at least one other
        if (prod(rowSums(reach[clique, clique])) > 0) {
            return(clique)
        }
    }
    return(NULL)
}

generate_cpt <- function(ig) {
    #' Generate binary conditional probability tables
    #'
    #' Generate a binary conditional probability tables for a given graph.
    #' @param ig The graph of class igraph.
    #' @return A binary conditional probability table.

    nodes <- igraph::V(ig)$name
    # Assign CPDs
    dist <- list()
    for (node in nodes) {
        parents <- igraph::neighbors(ig, node, "in")$name
        # Random number of states for the variable
        if (length(parents) == 0) { # No parents
            probs <- runif(2)
            probs <- probs / sum(probs)
            dist[[node]] <- matrix(probs, ncol = 2)
        } else { # Has parents
            card <- rep(2, length(parents))
            probs <- matrix(runif(prod(card) * 2), nrow = prod(card))
            probs <- probs / rowSums(probs)
            cpt <- array(t(probs), dim = c(2, card))
            dist[[node]] <- cpt
        }
    }
    return(dist)
}

# Main function
generate_data <- function(nodes, exp_degree, max_degree, targets,
                          seed = 0, connected = TRUE, samples_num = 0,
                          identifiable = FALSE, min_adj_size = 0,
                          discrete = FALSE) {
    #' Generate data
    #'
    #' Generate a causal graph and corresponding data given the parameters.
    #' @param nodes The number of nodes in the graph.
    #' @param exp_degree The expected degree of each node.
    #' @param max_degree The maximum degree of each node.
    #' @param targets The number of target nodes.
    #' @param seed The random seed. Default is 0.
    #' @param connected Whether the graph should be connected. Default is TRUE.
    #' @param samples_num The number of samples to generate. Default is 0.
    #' @param identifiable Whether the causal effects between the targets
    #' should be identifiable. Default is FALSE.
    #' @param min_adj_size The minimum size of the adjustment set required
    #' to be considered identifiable. Default is 0.
    #' @param discrete Whether the data should be discrete. Default is FALSE.
    #' @return A list with the sufficient statistics, targets, ID,
    #' and conditional probability table if discrete data is generated.

    set.seed(seed)
    n_targets <- targets
    while (TRUE) {
        # Sample topologically sorted DAG
        ig <- generate_dag(nodes, exp_degree, max_degree, connected)
        if (identifiable) { # Check if the graph is identifiable
            targets <- get_identifiable_targets(
                as(igraph::as_adjacency_matrix(ig), "graphNEL"),
                n_targets, min_adj_size
            )
        } else {
            targets <- sample(nodes, n_targets)
        }
        if (!is.null(targets)) break
    }
    # Shuffle node labels
    labels <- sample(nodes)
    # Sample edge weights
    w <- runif(nodes**2, 0.5, 3) * (rbinom(nodes**2, 1, 0.5) * 2 - 1)
    amat <- igraph::as_adjacency_matrix(ig) * matrix(w, nrow = nodes)
    data <- matrix(0, 2, nodes)
    # Generate linear Gaussian data for topologically sorted DAG
    if (samples_num > 0 && !discrete) {
        data <- pcalg::rmvDAG(samples_num, as(amat, "graphNEL"))
        # Adjust data to shuffled node labels
        data <- data[, match(seq_len(nodes), labels)]
        colnames(data) <- as.character(seq_len(nodes))
        cor <- cor(data)
    }
    # Adjust graph to shuffled node labels
    ig <- igraph::graph_from_adjacency_matrix(amat, "directed", TRUE)
    ig <- igraph::permute(ig, labels)
    # Generate discrete data for shuffled DAG
    if (samples_num > 0 && discrete) {
        igraph::V(ig)$name <- as.character(igraph::V(ig))
        cpt <- generate_cpt(ig)
        bnfit <- bnlearn::custom.fit(bnlearn::as.bn(ig), dist = cpt)
        data <- bnlearn::rbn(bnfit, n = samples_num)
        data <- matrix(match(as.matrix(data), LETTERS) - 1, ncol = nodes)
    }
    amat <- igraph::as_adjacency_matrix(ig, attr = "weight")
    dag <- as(amat, "graphNEL")
    # Adjust targets to shuffled node labels
    targets <- sort(dag@nodes[labels][targets])

    suffStat <- list(
        g = dag, jp = RBGL::johnson.all.pairs.sp(dag), # pcalg::dsepTest
        C = cor, n = samples_num, # pcalg::gaussCItest
        dm = data, adaptDF = TRUE, # pcalg::disCItest
        dagitty = pcalg::pcalg2dagitty(t(as(dag, "matrix")), dag@nodes)
    )
    return(list(
        "suffStat" = suffStat,
        "targets" = targets,
        "id" = digest::digest(c(dag, data, targets)),
        "cpt" = if (discrete) cpt else NULL
    ))
}

bng2graphNEL <- function(bng) {
    #' Bnlearn to graphNEL
    #'
    #' Convert the graph in bnlearn object to a graphNEL object.
    #' @param bng The bnlearn object.
    #' @return The graphNEL object.

    if (is.null(bng[[1]]$coefficients)) {
        return(bnlearn::as.graphNEL(bng))
    }
    amat <- bnlearn::amat(bng)
    for (i in seq_len(ncol(amat))) {
        amat[, i][bng[[i]]$parents] <- bng[[i]]$coefficients[-1]
    }
    ig <- igraph::graph_from_adjacency_matrix(amat, "directed", TRUE)
    amat <- igraph::as_adjacency_matrix(ig, attr = "weight")
    dag <- as(amat, "graphNEL")
    return(dag)
}

generate_data_from_file <- function(file, targets, seed = 0, samples_num = 0,
                                    identifiable = FALSE, min_adj_size = 0,
                                    ...) {
    #' Generate data from file
    #'
    #' Generate experimental data according to a model from bnlearn.
    #' @param file The filepath where the bnlearn object is stored.
    #' @param targets The number of target nodes.
    #' @param seed The random seed. Default is 0.
    #' @param samples_num The number of samples to generate. Default is 0.
    #' @param identifiable Whether the causal effects between the targets
    #' should be identifiable. Default is FALSE.
    #' @param min_adj_size The minimum size of the adjustment set required
    #' to be considered identifiable. Default is 0.
    #' @return A list with the sufficient statistics, targets and ID.

    set.seed(seed)
    bng <- readRDS(file)
    dag <- bng2graphNEL(bng)
    n_V <- length(bnlearn::nodes(bng))
    if (samples_num > 0) {
        data <- bnlearn::rbn(bng, n = samples_num)
        data <- data.matrix(data)
    } else {
        data <- matrix(0, 2, n_V)
    }
    if (identifiable) { # Check if the graph is identifiable
        amat <- bnlearn::amat(bng)
        ig <- igraph::graph_from_adjacency_matrix(amat, "directed")
        targets <- get_identifiable_targets(
            as(igraph::as_adjacency_matrix(ig), "graphNEL"),
            targets, min_adj_size
        )
    } else {
        targets <- sample(n_V, targets)
    }
    suffStat <- list(g = dag, dm = data)
    return(list(
        "suffStat" = suffStat,
        "targets" = sort(targets),
        "id" = digest::digest(c(dag, data, targets)),
        "cpt" = NULL
    ))
}


generate_discrete_samples_from_graph <- function(amat, targets, seed = 0,
                                                 samples_num) {
    #' Generate discrete experiment from graph
    #'
    #' Generate experimental data with discrete variables according to a graph.
    #' This function is used to generate discrete experiments for the same
    #' graphs that were used for linear Gaussian experiments.
    #' @param amat The adjacency matrix of the graph.
    #' @param targets The target nodes.
    #' @param seed The random seed. Default is 0.
    #' @param samples_num The number of samples to generate.

    set.seed(seed)
    nodes <- nrow(amat)
    ig <- igraph::graph_from_adjacency_matrix(amat, "directed", TRUE)

    igraph::V(ig)$name <- as.character(igraph::V(ig))
    cpt <- generate_cpt(ig)
    bnfit <- bnlearn::custom.fit(bnlearn::as.bn(ig), dist = cpt)
    data <- bnlearn::rbn(bnfit, n = samples_num)
    data <- matrix(match(as.matrix(data), LETTERS) - 1, ncol = nodes)

    amat <- igraph::as_adjacency_matrix(ig, attr = "weight")
    dag <- as(amat, "graphNEL")
    suffStat <- list(g = dag, dm = data)
    return(list(
        "suffStat" = suffStat,
        "targets" = targets,
        "id" = digest::digest(c(dag, data, targets)),
        "cpt" = cpt
    ))
}

generate_samples_from_graph <- function(amat, targets, seed = 0, samples_num,
                                        cpt = NULL) {
    #' Generate samples from graph
    #'
    #' Generate samples according to a graph.
    #' @param amat The adjacency matrix of the graph.
    #' @param targets The target nodes.
    #' @param seed The random seed. Default is 0.
    #' @param samples_num The number of samples to generate.
    #' @param cpt The conditional probability table. Default is NULL.
    #' @return A list with the data, ID and conditional probability table.

    set.seed(seed)
    nodes <- nrow(amat)
    if (is.null(cpt)) {
        ig <- igraph::graph_from_adjacency_matrix(amat, "directed", TRUE)
        # Sort nodes topologically
        topo_labels <- igraph::topo_sort(ig)
        topo_labels_order <- order(topo_labels)
        ig <- igraph::permute(ig, topo_labels_order)
        amat <- igraph::as_adjacency_matrix(ig, attr = "weight")
        dag <- as(amat, "graphNEL")
        data <- pcalg::rmvDAG(samples_num, dag)
        # Adjust data to shuffled node labels
        data <- data[, match(seq_len(nodes), topo_labels)]
    } else {
        dag <- as(amat != 0, "graphNEL")
        g <- bnlearn::as.bn(dag)
        bnfit <- bnlearn::custom.fit(g, dist = cpt)
        data <- bnlearn::rbn(bnfit, n = samples_num)
        data <- matrix(match(as.matrix(data), LETTERS) - 1, ncol = nodes)
    }
    colnames(data) <- as.character(seq_len(nodes))
    return(list(
        "dm" = data,
        "id" = digest::digest(c(dag, data, targets)),
        "cpt" = NULL
    ))
}
