from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import UCSepset, Meek
import networkx as nx
import numpy as np

from algorithms.snap import FastCausalGraph, skeleton_step


def pc(
    data: np.ndarray,
    ci_test: CIT,
    alpha: float,
    ignore: list[int] = [],
    oracle: bool = False,
    **kwargs,
) -> dict:
    """
    PC algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (CIT): CI test taking x, y and a conditioning set, and returning a p-value.
        It should provide a `method` attribute, like CIT in causal-learn.
        alpha (float): Significance level.
        ignore (list[int]): Nodes to ignore.
        oracle (bool): Whether d-separation tests are used.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned CPDAG.
    """
    # Skeleton search
    skeleton = FastCausalGraph(no_of_var=data.shape[1])
    skeleton.set_ind_test(ci_test)
    skeleton.G.graph[ignore, :] = skeleton.G.graph[:, ignore] = 0
    order = 0
    while skeleton.max_degree() > order:
        skeleton = skeleton_step(data, order, skeleton, alpha)
        order += 1

    # Orient v-structures
    if oracle:  # decide prioritization to get CPDAG
        pdag = UCSepset.uc_sepset(skeleton, 0)
    else:
        pdag = UCSepset.uc_sepset(skeleton, 3)
        pdag.G.reconstitute_dpath(pdag.G.get_graph_edges())

    # Orient Meek rules
    cpdag = Meek.meek(pdag)

    cpdag.to_nx_graph()
    amat = nx.to_numpy_array(cpdag.nx_graph).tolist()
    return {"amat": amat}
