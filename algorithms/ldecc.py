from itertools import permutations
from json import dumps
from typing import Callable

import numpy as np
import networkx as nx
import pandas as pd

from ldecc_github.ldecc import LDECCAlgorithm


def estimate_ancestry(
    results: dict,
    ci_test: Callable,
    alpha: float,
    num_nodes: int,
    targets: list[int],
) -> np.ndarray:
    """
    Generate adjacency matrix from local discovery results.

    Args:
        results (dict): The results of local discoveries.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        num_nodes (int): The number of nodes.
        targets (list[int]): The target nodes.

    Returns:
        np.ndarray: The estimated ancestries between targets.
    """
    amat = np.zeros((num_nodes, num_nodes))
    for t, result in results.items():
        amat[list(result["tmt_parents"]), t] = 1
        amat[t, list(result["tmt_children"])] = 1
        amat[list(result["unoriented"]), t] = 1
        amat[t, list(result["unoriented"])] = 1

    for t1, t2 in permutations(targets, 2):
        if amat[t1, t2] == 1 or amat[t2, t1] == 1:
            continue
        elif ci_test(t1, t2, list(results[t1]["tmt_parents"])) < alpha:
            amat[t1, t2] = 1
    return amat


def ldecc(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    targets: list[int],
    ignore: list[int] = [],
    oracle: bool = False,
    true_dag: nx.DiGraph | None = None,
    **kwargs,
) -> dict:
    """
    LDECC algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        ignore (list[int]): Nodes to ignore.
        oracle (bool): Whether to use CI oracle.
        true_dag (nx.DiGraph | None): The true causal DAG.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned causal relationships.
    """

    data = pd.DataFrame(data)
    results = {}
    for t in targets:
        ldecc_alg = LDECCAlgorithm(
            indep_test=ci_test,
            treatment_node=t,
            outcome_node=0,  # This does not matter
            alpha=alpha,
            use_ci_oracle=oracle,
            graph_true=true_dag,
            ignore=ignore,
        )
        result = ldecc_alg.run(data)
        for key, value in result.items():
            if isinstance(value, set):
                result[key] = list(value)
        for key, value in result["non_colliders"].items():
            if isinstance(value, set):
                result["non_colliders"][key] = list(value)
        results[int(t)] = result

    amat = estimate_ancestry(results, ci_test, data.shape[1], targets)
    return {"amat": amat.tolist(), "results": dumps(results)}
