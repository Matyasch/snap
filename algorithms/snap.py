from copy import deepcopy
from itertools import combinations

from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import UCSepset, Meek
from causallearn.utils.PCUtils.Helper import append_value
import networkx as nx
import numpy as np

from utils import FastCausalGraph
from causallearn.graph.GraphClass import CausalGraph


def skeleton_step(
    data: np.ndarray, order: int, cg: CausalGraph, alpha: float
) -> CausalGraph:
    """
    Skeleton step at a given order

    Args:
        data (np.ndarray): The data matrix.
        order (int): The order of CI tests.
        cg (CausalGraph): The current skeleton.
        alpha (float): Significance level.

    Returns:
        CausalGraph: The updated (in-place) skeleton.
    """
    for x in range(data.shape[1]):
        Neigh_x = cg.neighbors(x)
        if len(Neigh_x) < order - 1:
            continue
        for y in Neigh_x:
            curr_neigh = cg.neighbors(x)
            Neigh_x_noy = np.delete(curr_neigh, np.where(curr_neigh == y))
            for S in combinations(Neigh_x_noy, order):
                if cg.ci_test(x, y, S) >= alpha:
                    cg.G.graph[x, y] = cg.G.graph[y, x] = 0
                    append_value(cg.sepset, x, y, S)
                    append_value(cg.sepset, y, x, S)
                    break
    return cg


def find_minimal_separating_set(
    x: int, y: int, S: set[int], cg: CausalGraph, alpha: float
) -> set[int]:
    """
    Find minimal separating set for x and y that is a subseteq of separating set S,
    by removing nodes from S one by one until no more nodes can be removed
    without making x and y dependent. Note, that the found minimal separating set
    is not necessarily minimum sized.

    Args:
        x (int): Node x.
        y (int): Node y.
        S (set[int]): The (possibly not minimal) separating set.
        cg (CausalGraph): The causal graph.
        alpha (float): Significance level.

    Returns:
        set[int]: The minimal separating set.
    """
    min_S = S
    for v in S:
        if cg.ci_test(x, y, S - {v}) >= alpha:  # v can be removed
            min_S = find_minimal_separating_set(x, y, S - {v}, cg)
            break
    return min_S


def orient_v_structures_rfci(
    skeleton: CausalGraph, alpha: float
) -> tuple[CausalGraph, CausalGraph]:
    """
    Algorithm 4.4 Orienting v-structures in the RFCI algorithm (Colombo et al., 2012)

    Args:
        skeleton (CausalGraph): The skeleton to orient.
        alpha (float): Significance level.

    Returns:
        tuple[CausalGraph, CausalGraph]: The PDAG and the new skeleton.
    """
    M = [(i, j, k) for (i, j, k) in skeleton.find_unshielded_triples() if i < k]
    L = []
    while len(M) > 0:
        x, y, z = M.pop(0)
        Sxz = skeleton.sepset[x, z][0]
        if y in Sxz:
            continue
        indxy = skeleton.ci_test(x, y, Sxz) >= alpha
        indyz = skeleton.ci_test(y, z, Sxz) >= alpha
        # both dependent
        if not indxy and not indyz:
            L.append((x, y, z))
        else:
            for r, ind in [(x, indxy), (z, indyz)]:
                if ind:
                    # find minimal separating set
                    Sry = find_minimal_separating_set(r, y, set(Sxz), skeleton, alpha)
                    # Add Y as sepset
                    append_value(skeleton.sepset, r, y, Sry)
                    append_value(skeleton.sepset, y, r, Sry)
                    # Add triples that form a triangle with Xr-Xy
                    triangles = skeleton.find_triangles()
                    for i, j, k in triangles:
                        if i == r and k == y or i == y and k == r:
                            M.append((min(i, k), j, max(i, k)))
                    # Delete triples containing Xr-Xy from M and L
                    for i, j, k in M:
                        if (
                            (i == r and j == y)
                            or (i == r and j == r)
                            or (j == r and k == y)
                            or (j == y and k == r)
                        ):
                            M.remove((i, j, k))
                    for i, j, k in L:
                        if (
                            (i == r and j == y)
                            or (i == r and j == r)
                            or (j == r and k == y)
                            or (j == y and k == r)
                        ):
                            L.remove((i, j, k))
                    # Delete edge
                    skeleton.G.graph[r, y] = skeleton.G.graph[y, r] = 0
    pdag = deepcopy(skeleton)
    for x, y, z in L:
        pdag.G.graph[y, x] = pdag.G.graph[y, z] = 1
    return pdag, skeleton


def get_def_non_ancestors(cg: CausalGraph, targets: list[int]) -> list[int]:
    """
    Get nodes that are definite non-ancestors of any target

    Args:
        cg (CausalGraph): The causal graph.
        targets (list[int]): The target nodes.

    Returns:
        list[int]: Definite non-ancestors of any target.
    """
    amat = cg.G.graph
    amat = -amat.copy()
    amat[amat == -1] = 0
    np.fill_diagonal(amat, 1)
    amat = amat.astype(bool)
    # Reachability matrix
    reach = np.linalg.matrix_power(amat, amat.shape[0] - 1)
    # Nodes that do not reach any target
    non_anc = np.all(np.invert(reach[:, targets]), axis=1)
    return np.where(non_anc)[0]


def snap(
    data: np.ndarray,
    ci_test: CIT,
    alpha: float,
    targets: list[int],
    ignore: list[int] = [],
    max_order: int = -1,
    filter_mode: bool = False,
    oracle: bool = False,
    **kwargs,
) -> list[int] | dict:
    """
    SNAP algorithm to find causal graph or the non-ancestors of targets

    Args:
        data (np.ndarray): The data matrix.
        ci_test (CIT): CI test taking x, y and a conditioning set, and returning a p-value.
        It should provide a `method` attribute, like CIT in causal-learn.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        ignore (list[int]): Nodes to ignore.
        max_order (int): The maximum order of CI tests.
        filter_mode (bool): Whether SNAP is only used to find non-ancestors.
        oracle (bool): Whether d-separation tests are used.
        **kwargs: Additional arguments are ignored.

    Returns:
        list[int] | dict: Non-ancestors or adjacency matrix of CPDAG.
    """
    if max_order < 0 and filter_mode:
        raise ValueError("Max order must be specified for filtering mode")
    elif max_order < 0 and not filter_mode:
        max_order = data.shape[1] - 1  # Run until completion

    skeleton = FastCausalGraph(no_of_var=data.shape[1])
    skeleton.set_ind_test(ci_test)
    skeleton.G.graph[ignore, :] = skeleton.G.graph[:, ignore] = 0

    non_anc = []
    for order in range(max_order + 1):
        if skeleton.max_degree() <= order:
            break
        skeleton = skeleton_step(data, order, skeleton, alpha)
        if order < 2:
            g = UCSepset.uc_sepset(skeleton, 1)
        else:
            g, skeleton = orient_v_structures_rfci(skeleton, alpha)
        non_anc = get_def_non_ancestors(g, targets)
        skeleton.G.graph[non_anc, :] = skeleton.G.graph[:, non_anc] = 0

    if filter_mode:
        return {"non_anc": non_anc}

    # Early stopped due to max order
    if max_order < 2:
        pdag = UCSepset.uc_sepset(skeleton, 1)
    elif max_order < data.shape[1] - 1:
        pdag = orient_v_structures_rfci(skeleton, alpha)[0]
    # Ran until completion, decide v-structure prioritization to get CPDAG
    elif oracle:
        pdag = UCSepset.uc_sepset(skeleton, 0)
    else:
        pdag = UCSepset.uc_sepset(skeleton, 3)

    # Orient Meek rules
    cpdag = Meek.meek(pdag)
    non_anc = get_def_non_ancestors(cpdag, targets)
    cpdag.G.graph[non_anc, :] = cpdag.G.graph[:, non_anc] = 0

    cpdag.to_nx_graph()
    amat = nx.to_numpy_array(cpdag.nx_graph)
    return {"amat": amat}
