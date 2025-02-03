from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import combinations, permutations
from json import loads
from typing import Callable

import networkx as nx
import numpy as np
from rpy2.robjects import default_converter, globalenv, numpy2ri, r

from ldecc_github.utils import get_all_combinations
from utils import generate_data

r.source("evaluate/evaluate.R")
r.source("generate_data.R")


def get_test_nums(results: list[dict]) -> np.ndarray:
    """
    Get the number of tests performed by the discovery algorithm in each experiment.

    Args:
        results (list[dict]): List of result dictionaries from experiments.

    Returns:
        np.ndarray: Array of CI test numbers.
    """
    test_num = [np.sum(res["tests"]) for res in results]
    return np.array(test_num)


def get_filter_test_nums(results: list[dict]) -> np.ndarray:
    """
    Get the number of tests performed by pre-filtering in each experiment.

    Args:
        results (list[dict]): List of result dictionaries from experiments.

    Returns:
        np.ndarray: Array of CI test numbers.
    """
    test_sum = [np.sum(res["filter_tests"]) for res in results]
    return np.array(test_sum)


def get_test_per_order(results: list[dict]) -> np.ndarray:
    """
    Get the number of tests performed by pre-filtering and the discovery algorithm
    per order of adjustment set for each experiment.
    The order is the index of the array.

    Args:
        results (list[dict]): List of result dictionaries from experiments.

    Returns:
        np.ndarray: Array of CI test numbers at each order.
    """
    test_num = [
        np.array(res["tests"]) + np.array(res["filter_tests"]) for res in results
    ]
    return np.array(test_num)


def get_times(results: list[dict]) -> np.ndarray:
    """
    Get the running time of each experiment.

    Args:
        results (list[dict]): List of result dictionaries from experiments.

    Returns:
        np.ndarray: Array running times.
    """
    return np.array([res["time"] for res in results])


def get_experiments(
    nodes: int,
    seed: int = 0,
    exp: int = 100,
    file: str | None = None,
    exp_degree: float = 3.0,
    max_degree: int = 10,
    targets: int = 4,
    connected: bool = True,
    identifiable: bool = False,
    min_adj_size: int = 0,
    samples: int = 1000,
    discrete: bool = False,
) -> dict:
    """
    Generate experimental data for a set of parameters.

    Args:
        nodes (int): Number of nodes in the graph.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        exp (int, optional): Number of experiments to generate. Defaults to 100.
        file (str, optional): File path if data comes from bnlearn. Defaults to None.
        exp_degree (float, optional): Expected degree of nodes in the graph. Defaults to 3.0.
        max_degree (int, optional): Maximum degree of nodes in the graph. Defaults to 10.
        targets (int, optional): Number of target nodes. Defaults to 4.
        connected (bool, optional): Ensure the graph is connected. Defaults to True.
        identifiable (bool, optional): Ensure the causal effects are identifiable. Defaults to False.
        min_adj_size (int, optional): Minimum size of adjustment sets. Defaults to 0.
        samples (int, optional): Number of samples to generate. Defaults to 1000.
        discrete (bool, optional): Generate discrete data. Defaults to False.

    Returns:
        dict: Dictionary containing the generated experimental data, identified by their ids.
    """
    proc = []
    with ProcessPoolExecutor() as executor:
        for s in range(exp):
            # Generate data
            p = executor.submit(
                generate_data,
                seed=seed + s,
                file=file,
                nodes=nodes,
                exp_degree=exp_degree,
                max_degree=max_degree,
                targets=targets,
                connected=connected,
                identifiable=identifiable,
                min_adj_size=min_adj_size,
                samples_num=samples,
                discrete=discrete,
            )
            proc.append(p)
    experiments = {}
    for p in as_completed(proc):
        result = p.result()
        experiments[result["id"]] = result
    return experiments


def generate_samples_for_experiments(
    experiments: dict, samples_num: int = 1000, seed=100
) -> dict:
    """
    Generate new samples for each experiment.

    Args:
        experiments (dict): Dictionary containing the experimental data.
        samples_num (int, optional): Number of samples to generate. Defaults to 1000.
        seed (int, optional): Random seed for reproducibility. Defaults to 100.

    Returns:
        dict: Dictionary containing the generated samples for each experiment, identified by their ids..
    """
    samples = {}
    for exp_id, exp in experiments.items():
        with (default_converter + numpy2ri.converter).context():
            dm = r["generate_samples_from_graph"](
                amat=nx.to_numpy_array(exp["true_dag"]),
                targets=np.array(exp["targets"]) + 1,
                seed=seed,
                samples_num=samples_num,
                cpt=exp["cpt"],
            )["dm"]
            samples[exp_id] = np.array(dm)
    return samples


def get_shd(est_amat: np.ndarray, ora_amat: np.ndarray, targets: list[int]) -> int:
    """
    Compute the Structural Hamming Distance (SHD) between an estimated and a true adjacency matrix,
    restricted to the possible ancestors of the target variables.

    Args:
        est_amat (np.ndarray): Estimated adjacency matrix.
        ora_amat (np.ndarray): True adjacency matrix.
        targets (list[int]): List of target variables.

    Returns:
        int: The Structural Hamming Distance.
    """
    with (default_converter + numpy2ri.converter).context():
        cpdag = r["dag2cpdag"](r["as"](ora_amat != 0, "graphNEL"))
        ora_amat = np.array(r["as"](cpdag, "matrix")).astype(np.int8)

        # Get non-ancestors of targets
        reach = ora_amat.copy().astype(bool)
        np.fill_diagonal(reach, True)
        reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)[:, targets]
        # Nodes that do not reach any target
        non_anc = np.all(np.invert(reach), axis=1)
        # Only consider graphs over possible ancestors of targets
        ora_amat[non_anc, :] = ora_amat[:, non_anc] = 0
        est_amat[non_anc, :] = est_amat[:, non_anc] = 0
        return r["shd"](ora_amat, est_amat)[0]


def get_shds(results: list[dict], experiments: dict) -> np.ndarray:
    """
    Compute the Structural Hamming Distance (SHD) between the estimated and true adjacency matrices,
    restricted to the possible ancestors of the target variables, for each experiment.

    Args:
        results (list[dict]): List of result dictionaries from experiments.
        experiments (dict): Dictionary containing the true experimental data.

    Returns:
        np.ndarray: Array of SHD values.
    """
    proc = []
    with ProcessPoolExecutor() as e:
        for res in results:
            targets = experiments[res["id"]]["targets"]
            est_amat = np.array(res["amat"]).astype(np.int8)
            ora_amat = nx.to_numpy_array(experiments[res["id"]]["true_dag"])
            p = e.submit(get_shd, est_amat, ora_amat, targets)
            proc.append(p)

        shds = []
        for p in as_completed(proc):
            shds.append(p.result())
    return np.array(shds)


def is_ancestor(amat: np.ndarray, t1: int, t2: int) -> bool:
    """
    Check if t1 is an (possible) ancestor of t2 in the graph defined by the adjacency matrix.

    Args:
        amat (np.ndarray): Adjacency matrix representing the graph.
        t1 (int): Index of the first node.
        t2 (int): Index of the second node.

    Returns:
        bool: True if t1 is an (possible) ancestor of t2, False otherwise.
    """
    reach = amat.copy().astype(bool)
    np.fill_diagonal(reach, True)
    reach = np.linalg.matrix_power(reach, reach.shape[0] - 1)
    return reach[t1, t2]


def get_aid(results: list[dict], experiments: dict, aid_func: Callable) -> np.ndarray:
    """
    Compute the Adjustment Identification Distance (AID) for each experiment.

    Args:
        results (list[dict]): List of result dictionaries from experiments.
        experiments (dict): Dictionary containing the true experimental data.
        aid_func (Callable): Identification distance function.

    Returns:
        np.ndarray: Array of AID values.
    """
    est_amats = {res["id"]: np.array(res["amat"]).astype(np.int8) for res in results}

    distances = []
    for exp_id, exp in experiments.items():
        targets = exp["targets"]
        ora_amat = nx.to_numpy_array(exp["true_dag"]) != 0
        ora_amat = ora_amat.astype(np.int8)
        est_amat = est_amats[exp_id]
        # gadjid format
        ora_amat[ora_amat == ora_amat.T] *= 2
        est_amat[est_amat == est_amat.T] *= 2
        try:
            distance = aid_func(
                ora_amat,
                est_amat,
                treatments=targets,
                effects=targets,
                edge_direction="from row to column",
            )
            distances.append(distance[1])
        except Exception:
            distances.append(len(targets) * (len(targets) - 1))
    return np.array(distances)


def true_linear_gaussian_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, **kwargs
) -> float:
    """
    Compute the true causal effect between two nodes in a linear Gaussian model.

    Args:
        treatment (int): Index of the treatment node.
        outcome (int): Index of the outcome node.
        true_dag (nx.DiGraph): The true causal DAG with edge weights representing the coefficients.

    Returns:
        float: The true causal effect of the treatment on the outcome.
    """
    amat = nx.to_numpy_array(true_dag)
    return sum(
        np.prod([amat[path[i], path[i + 1]] for i in range(len(path) - 1)])
        for path in nx.all_simple_paths(true_dag, treatment, outcome)
    )


def true_binary_effect(
    treatment: int, outcome: int, true_dag: nx.DiGraph, cpt: object, **kwargs
) -> float:
    """
    Compute the true causal effect between two nodes in a model of binary variables.

    Args:
        treatment (int): Index of the treatment node.
        outcome (int): Index of the outcome node.
        true_dag (nx.DiGraph): The true causal DAG.
        cpt (object): The conditional probability table.

    Returns:
        float: The true causal effect of the treatment on the outcome.
    """
    if treatment not in nx.ancestors(true_dag, outcome):
        return 0.0
    with (default_converter + numpy2ri.converter).context():
        return globalenv["true_binary_effect"](
            int(treatment) + 1, int(outcome) + 1, nx.to_numpy_array(true_dag), cpt
        )[0]


def get_true_causal_effects(experiments: dict, family="gaussian") -> dict:
    """
    Compute the true causal effects between all pairs of target nodes in each experiment.

    Args:
        experiments (dict): Dictionary containing the experimental data.
        family (str, optional): The family of the model ('gaussian' or 'binary'). Defaults to 'gaussian'.

    Returns:
        dict: Dictionary containing the true causal effects between each pair of targets
        for each experiment, identified by their ids.
    """
    if family == "gaussian":
        get_effect = true_linear_gaussian_effect
    elif family == "binary":
        get_effect = true_binary_effect
    else:
        raise ValueError("Invalid family")
    effects = {}
    for exp_id, exp in experiments.items():
        effects[exp_id] = {}
        for t1, t2 in permutations(exp["targets"], 2):
            effects[exp_id][(t1, t2)] = get_effect(t1, t2, **exp)
    return effects


def get_local_ida_adj_sets(
    target: int,
    amat: np.ndarray | None = None,  # full cpdag
    results: dict | None = None,  # local results
) -> list[list[int]]:
    """
    Compute adjustment sets with the local IDA algorithm given a full CPDAG
    or results from a local causal discovery algorithm.

    Args:
        target (int): Index of the target node.
        amat (np.ndarray | None, optional): Full CPDAG adjacency matrix. Defaults to None.
        results (dict | None, optional): Local results. Defaults to None.

    Returns:
        list: List of local adjustment sets.
    """
    if amat is None:  # local results
        parents = list(results[target]["tmt_parents"])
        valid_sets = get_all_combinations(
            results[target]["unoriented"], results[target]["non_colliders"]
        )
        return [parents + unoriented for unoriented in valid_sets]
    else:  # full cpdag
        parents = np.logical_and(amat[:, target] != 0, amat[target, :] == 0)
        parents = np.where(parents)[0].tolist()
        unoriented = np.logical_and(amat[:, target] != 0, amat[target, :] != 0)
        unoriented = np.where(unoriented)[0]
        skeleton = (amat + amat.T) != 0
        np.fill_diagonal(skeleton, True)

        valid_sets = [parents]
        for l in range(1, len(unoriented) + 1):
            for comb in combinations(unoriented, l):
                candidate_set = list(comb) + parents
                if np.all(skeleton[comb, :][:, candidate_set]):
                    valid_sets.append(candidate_set)
        return valid_sets


def convert_local_results(results: str) -> dict:
    """
    Convert results from a local causal discovery algorithm from a JSON string into a dictionary.

    Args:
        results (str): JSON string containing the results.

    Returns:
        dict: Dictionary containing the converted results.
    """
    results = loads(results)
    new_results = {int(key): {} for key in results}
    for key, value in results.items():
        new_res = {}
        new_res["tmt_parents"] = set(value["tmt_parents"])
        new_res["tmt_children"] = set(value["tmt_children"])
        new_res["unoriented"] = set(value["unoriented"])
        new_res["non_colliders"] = {
            int(key): set(value) for key, value in value["non_colliders"].items()
        }
        new_results[int(key)] = new_res
    return new_results


def get_adj_sets(
    amat: np.ndarray,
    treatment: int,
    outcome: int,
    scope: str,
    results: dict | None = None,
) -> list[list[int]]:
    """
    Compute adjustment sets for a given treatment and outcome pair in a graph defined by an adjacency matrix.
    If the causal effect is unidentifiable, fall back to the local IDA algorithm.

    Args:
        amat (np.ndarray): Adjacency matrix representing the graph.
        treatment (int): Index of the treatment node.
        outcome (int): Index of the outcome node.
        scope (str): Whether the adjacencmy matrix is from a 'global' or 'local' algorithm.
        results (dict, optional): Results of a local algorithm. Defaults to None.

    Returns:
        list: List of adjustment sets.
    """
    if not is_ancestor(amat, treatment, outcome):
        return [0.0]
    if scope == "global":
        with (default_converter + numpy2ri.converter).context():
            adj_set = r["get_adjustmet_set"](treatment + 1, outcome + 1, amat != 0)
        if adj_set.__class__ == np.ndarray:  # identifiable
            adj_sets = [(np.array(adj_set) - 1).tolist()]
        else:  # unidentifiable, use local IDA
            adj_sets = get_local_ida_adj_sets(treatment, amat=amat)
    elif scope == "local":  # local results, use local IDA
        adj_sets = get_local_ida_adj_sets(treatment, results=results)
    else:
        raise ValueError("Invalid scope")
    return adj_sets


def estimate_binary(
    samples: np.ndarray, treatment: int, outcome: int, adj_set: list[int], val: int
) -> float:
    """
    Compute E[outcome | do(treatment = val)] = E[outcome | treatment = val, adj_set]
    empirically from the samples.

    Args:
        samples (np.ndarray): The empirical samples.
        val (int): Value of the treatment to intervene on.
        outcome (int): Index of the outcome node.
        treatment (int): Index of the treatment node.
        adj_set (list[int]): List of adjustment set nodes.

    Returns:
        float: The estimated expectation of the outcome.
    """
    # Filter rows based on the treatment value
    samples_val = samples[samples[:, treatment] == val]
    # Create the design matrix X
    X = np.column_stack(
        [np.ones(len(samples_val))] + [samples_val[:, idx] for idx in adj_set]
    )
    # Outcome variable (filtered by val)
    y = samples_val[:, outcome]
    # Solve the least squares problem X * beta = y
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    # Predict the outcome for all rows in the original samples using the estimated beta
    X_full = np.column_stack(
        [np.ones(len(samples))] + [samples[:, idx] for idx in adj_set]
    )
    y_pred = X_full @ beta
    return y_pred.mean()


def estimate_causal_effect(
    treatment: int,
    outcome: int,
    samples: np.ndarray,
    amat: np.ndarray,
    scope: str = "global",
    family: str = "gaussian",
    results: dict | None = None,
) -> list[float]:
    """
    Estimate the causal effect of a treatment on an outcome given samples and a graph.

    Args:
        treatment (int): Index of the treatment node.
        outcome (int): Index of the outcome node.
        samples (np.ndarray): The empirical samples.
        amat (np.ndarray): Adjacency matrix representing the graph.
        scope (str, optional): Whether the adjacency matrix is from a 'global' or 'local' algorithm.
        Defaults to 'global'.
        family (str, optional): The family of the model ('gaussian' or 'binary'). Defaults to 'gaussian'.
        results (dict, optional): Results of a local algorithm. Defaults to None.

    Returns:
        list[float]: List of estimated causal effects.
    """
    if not is_ancestor(amat, treatment, outcome):
        return [0.0]
    adj_sets = get_adj_sets(amat, treatment, outcome, scope, results)

    effects = []
    for adj_set in adj_sets:
        if family == "gaussian":
            x = samples[:, [treatment] + adj_set]
            y = samples[:, outcome]
            A = np.vstack((x.T, np.ones(len(x))))
            effect = np.linalg.lstsq(A.T, y, rcond=None)[0][0]
            effects.append(effect)
        elif family == "binary":
            do_1 = estimate_binary(samples, treatment, outcome, adj_set, 1)
            do_0 = estimate_binary(samples, treatment, outcome, adj_set, 0)
            effects.append(do_1 - do_0)
    return effects


def estimate_causal_effects(
    results: list,
    samples: dict,
    scope: str = "global",
    family: str = "gaussian",
) -> np.ndarray:
    """
    Estimate the causal effects between all pairs of target nodes based on the results of each experiment.

    Args:
        results (list): List of result dictionaries from experiments.
        samples (dict): Dictionary containing samples for each experiment.
        scope (str, optional): Whether the adjacency matrix is from a 'global' or 'local' algorithm.
        Defaults to 'global'.
        family (str, optional): The family of the model ('gaussian' or 'binary'). Defaults to 'gaussian'.

    Returns:
        dict: Dictionary containing the estimated causal effects for each experiment, identified by their ids.
    """
    ates = {}
    proc = []
    with ProcessPoolExecutor() as e:
        for res in results:
            ates[res["id"]] = {}
            amat = np.array(res["amat"]).astype(np.int8)
            dm = samples[res["id"]]
            if scope == "local":
                results = convert_local_results(res["results"])
            else:
                results = None
            for t1, t2 in permutations(res["targets"], 2):
                p = e.submit(
                    estimate_causal_effect,
                    treatment=t1,
                    outcome=t2,
                    samples=dm,
                    amat=amat,
                    scope=scope,
                    family=family,
                    results=results,
                )
                p.id = res["id"]
                p.t1 = t1
                p.t2 = t2
                proc.append(p)

        for p in as_completed(proc):
            ates[p.id][(p.t1, p.t2)] = p.result()
        return ates


def get_intervention_distance(est_ates: dict, true_ates: dict) -> np.ndarray:
    """
    Compute the Intervention Distance between the estimated and true causal effects.

    Args:
        est_ates (dict): Dictionary containing the estimated causal effects for each experiment.
        true_ates (dict): Dictionary containing the true causal effects for each experiment.

    Returns:
        np.ndarray: Array of intervention distances for each experiment.
    """
    distances = []
    for exp in est_ates:
        exp_dist = []
        for pair in est_ates[exp]:
            true_ate = true_ates[exp][pair]
            dist = [np.abs(true_ate - est_ate) for est_ate in est_ates[exp][pair]]
            exp_dist.append(np.mean(dist))
        distances.append(np.mean(exp_dist))
    return np.array(distances)
