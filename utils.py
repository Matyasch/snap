from collections import defaultdict
import json
from pathlib import Path
import pickle
from types import MethodType
from typing import Callable

from causallearn.graph.Edge import Edge
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.cit import CIT
import numpy as np
import networkx as nx
from rpy2.robjects import r

r.source("generate_data.R")


class CountingTest:
    """
    Wrapper for CI tests that counts the number of tests done.
    """

    def __init__(
        self,
        data: np.ndarray,
        ci_test: str,
        **kwargs,
    ):
        self.cit = CIT(data, ci_test, **kwargs)
        self.method = self.cit.method
        self.tests_done = defaultdict(set)

    def __call__(
        self, X: int, Y: int, condition_set: list[int] | None = [], *args, **kwargs
    ):
        if condition_set is None:
            condition_set = []
        self.tests_done[frozenset((X, Y))] |= {tuple(condition_set)}
        p = self.cit(X, Y, condition_set)
        return p

    def get_tests_per_order(self) -> np.ndarray:
        """
        Get the number of tests done per order.

        Returns:
            np.ndarray: The number of tests done per order.
        """
        num_nodes = self.cit.data.shape[1]
        cond_sets = self.tests_done.values()
        if not cond_sets:
            return np.zeros(num_nodes, dtype=int)
        orders, test_num = np.unique(
            [len(cond) for conds in cond_sets for cond in conds],
            return_counts=True,
        )
        tests_per_order = np.zeros(num_nodes, dtype=int)
        tests_per_order[orders] = test_num
        return tests_per_order


class FastCausalGraph(CausalGraph):
    """
    CausalGraph that implements faster edge removal and kite search.
    """

    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
        self.G.remove_edge = MethodType(FastCausalGraph._remove_edge, self.G)

    def _remove_edge(self, edge: Edge):
        """
        Same as the original function, but without the call to reconstitute_dpath and only considers CPDAGs.
        """
        i = self.node_map[edge.get_node1()]
        j = self.node_map[edge.get_node2()]
        self.graph[j, i] = 0
        self.graph[i, j] = 0

    @staticmethod
    def from_amat(amat: np.ndarray):
        """
        Create a FastCausalGraph from an adjacency matrix.

        Args:
            amat (np.ndarray): The adjacency matrix.

        Returns:
            FastCausalGraph: The created FastCausalGraph.
        """
        amat = amat.copy()
        amat[amat == 1] = -1
        amat[np.logical_and(amat == 0, amat.T == -1)] = 1
        cg = FastCausalGraph(len(amat))
        cg.G.graph = amat
        return cg


def save_data(data: dict, seed: int, **kwargs):
    """
    Save experimental data to a file identified by the seed and the kwargs.

    Args:
        data (dict): The experimental data to save.
        seed (int): The seed used to generate the data.
        **kwargs: Additional arguments used to generate the data.
    """
    file_path = Path("experiments/{}/{}.pkl".format(json.dumps(kwargs), seed))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as file:
        pickle.dump(data, file)


def load_data(seed: int, **kwargs) -> dict:
    """
    Load experimental data from a file identified by the seed and the kwargs.

    Args:
        seed (int): The seed used to generate the data.
        **kwargs: Additional arguments used to generate the data.

    Returns:
        dict: The loaded experimental data.
    """
    file_path = Path("experiments/{}/{}.pkl".format(json.dumps(kwargs), seed))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("rb") as file:
        return pickle.load(file)


def generate_data(
    seed: int,
    file: str | None,
    nodes: int,
    exp_degree: float,
    max_degree: int,
    targets: int,
    connected: bool,
    identifiable: bool,
    min_adj_size: int,
    samples_num: int,
    discrete: bool,
) -> dict:
    """
    Generate experimental data given the parameters. Generated data is also saved
    in the `experiments` directory and loaded if the same parameters are used again.

    Args:
        seed (int): The seed for the random number generator.
        file (str | None): The file to load data from.
        nodes (int): Number of nodes.
        exp_degree (float): Expected degree of the graph.
        max_degree (int): Maximum degree of the graph.
        targets (int): Number of target nodes.
        connected (bool): Whether the graph should be connected.
        identifiable (bool): Whether the graph should be identifiable.
        min_adj_size (int): Minimum size of the adjacency set.
        samples_num (int): Number of samples.
        discrete (bool): Whether the data is discrete.

    Returns:
        dict: The generated experimental data.
    """
    if file is None:  # Generate causal model from scratch
        assert nodes > 0
        gen_func = r["generate_data"]
        kwargs = {
            "seed": seed,
            "nodes": nodes,
            "exp_degree": exp_degree,
            "max_degree": max_degree,
            "targets": targets,
            "connected": connected,
            "identifiable": identifiable,
            "min_adj_size": min_adj_size,
            "samples_num": samples_num,
            "discrete": discrete,
        }
    else:  # Generate data according to model from file
        gen_func = r["generate_data_from_file"]
        kwargs = {
            "file": file,
            "seed": seed,
            "targets": targets,
            "identifiable": identifiable,
            "min_adj_size": min_adj_size,
            "samples_num": samples_num,
        }
    try:  # Try to load previously generated experimental data
        return load_data(**kwargs)
    except FileNotFoundError:  # Generate new experimental data
        r_data = gen_func(**kwargs)
        true_dag = r_data.rx2("suffStat").rx2("g")
        true_dag = nx.from_numpy_array(
            np.array(r["as"](true_dag, "matrix")),
            create_using=nx.DiGraph,
        )
        dm = np.array(r_data.rx2("suffStat").rx2("dm"))
        data = {
            "id": r_data.rx2("id")[0],
            "data": dm,
            "targets": np.array(r_data.rx2("targets")).astype(np.int32) - 1,
            "true_dag": true_dag,
            "cpt": r_data.rx2("cpt"),
        }
        save_data(data, **kwargs)
        return data
