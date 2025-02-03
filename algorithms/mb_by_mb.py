from json import dumps
from typing import Callable

import numpy as np
import networkx as nx
import pandas as pd

from algorithms.ldecc import estimate_ancestry
from ldecc_github.mb_by_mb import MBbyMBAlgorithm


def mb_by_mb(
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
    MB-by-MB algorithm

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
        mb_by_mb_alg = MBbyMBAlgorithm(
            indep_test=ci_test,
            treatment_node=t,
            outcome_node=0,  # This does not matter
            alpha=alpha,
            use_ci_oracle=oracle,
            graph_true=true_dag,
            ignore=ignore,
        )
        result = mb_by_mb_alg.run(data)
        for key, value in result.items():
            if isinstance(value, set):
                result[key] = list(value)
        for key, value in result["non_colliders"].items():
            if isinstance(value, set):
                result["non_colliders"][key] = list(value)
        results[int(t)] = result

    amat = estimate_ancestry(results, ci_test, data.shape[1], targets)
    return {"amat": amat.tolist(), "results": dumps(results)}
