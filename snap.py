from collections import defaultdict
from itertools import combinations
from typing import Callable

import numpy as np


def skeleton_step(
    order: int,
    g: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    sep_set: dict[frozenset, list],
) -> np.ndarray:
    """
    Skeleton step at a given order

    Args:
        order (int): The order of CI tests.
        g (np.ndarray): The current skeleton.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        sep_set (dict[frozenset, list]): The separating sets.

    Returns:
        np.ndarray: The updated (in-place) skeleton.
    """
    for x in range(len(g)):
        Neigh_x = np.where(g[x, :] != 0)[0]
        if len(Neigh_x) < order - 1:
            continue
        for y in Neigh_x:
            curr_neigh = np.where(g[x, :] != 0)[0]
            Neigh_x_noy = np.delete(curr_neigh, np.where(curr_neigh == y))
            for S in combinations(Neigh_x_noy, order):
                if ci_test(x, y, S) >= alpha:
                    g[x, y] = g[y, x] = 0
                    sep_set[frozenset((x, y))].append(S)
                    break
    return g


def find_unshielded_triples(g: np.ndarray) -> np.ndarray:
    """
    Find all unshielded triples in the skeleton in a vectorized way.

    Args:
        g (np.ndarray): The skeleton.

    Returns:
        np.ndarray: The unshielded triples.
    """
    i, j = np.where(g != 0)  # i - j
    k = np.where((g[j] != 0) & (g[i] == 0))  # j - k -/- i
    # Extract actual indices
    i, j, k = i[k[0]], j[k[0]], k[1]
    # Ensure no duplicates
    mask = i < k
    return np.column_stack((i[mask], j[mask], k[mask]))


def v_struct_pc(
    skeleton: np.ndarray, sep_set: dict[frozenset, list], bidirected: bool = False
) -> np.ndarray:
    """
    Orient v-structures in the skeleton as in the PC algorithm

    Args:
        skeleton (np.ndarray): The skeleton to orient.
        sep_set (dict[frozenset, list]): The separating sets.
        bidirected (bool): Whether to orient conflicts as bidirected, or overwrite.

    Returns:
        np.ndarray: The PDAG.
    """
    pdag = skeleton.copy()
    for x, y, z in find_unshielded_triples(skeleton):
        if all(y not in S for S in sep_set[frozenset((x, z))]):
            pdag[y, x] = pdag[y, z] = 1  # Orient arrowhead
            if not bidirected:
                pdag[x, y] = pdag[z, y] = -1  # Orient tail
    return pdag


def find_minimal_separating_set(
    x: int,
    y: int,
    S: set[int],
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
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
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.

    Returns:
        set[int]: The minimal separating set.
    """
    min_S = S
    for v in S:
        if ci_test(x, y, S - {v}) >= alpha:  # v can be removed
            min_S = find_minimal_separating_set(x, y, S - {v}, ci_test, alpha)
            break
    return min_S


def v_struct_rfci(
    skeleton: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    sep_set: dict[frozenset, list],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 4.4 Orienting v-structures in the RFCI algorithm (Colombo et al., 2012)

    Args:
        skeleton (np.ndarray): The skeleton to orient.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        sep_set (dict[frozenset, list]): The separating sets.

    Returns:
        tuple[np.ndarray, np.ndarray]: The PDAG and the new skeleton.
    """
    M = find_unshielded_triples(skeleton).tolist()
    L = []
    while len(M) > 0:
        xi, xj, xk = M.pop(0)
        S_xi_xk = sep_set[frozenset((xi, xk))][0]
        if xj in S_xi_xk:
            continue
        ind_xi_xj = ci_test(xi, xj, S_xi_xk) >= alpha
        ind_xj_xk = ci_test(xj, xk, S_xi_xk) >= alpha
        # both dependent
        if not ind_xi_xj and not ind_xj_xk:
            L.append((xi, xj, xk))
        else:
            for xr, ind in [(xi, ind_xi_xj), (xk, ind_xj_xk)]:
                if ind:
                    # find minimal separating set
                    S_xr_xj = find_minimal_separating_set(
                        xr, xj, set(S_xi_xk), ci_test, alpha
                    )
                    # Add Y as sepset
                    sep_set[frozenset((xr, xj))].append(S_xr_xj)
                    # Add triples of the form Xr - V - Xy
                    for v in np.where(((skeleton[xr] != 0) & (skeleton[xj] != 0)))[0]:
                        M.append((min(xr, xj), v, max(xr, xj)))
                    # Delete triples of the form Xr - Xy - V from M and L
                    for i, j, k in M:
                        if (i, j, k) in M and (
                            (i == xr and j == xj)
                            or (i == xj and j == xr)
                            or (j == xr and k == xj)
                            or (j == xj and k == xr)
                        ):
                            M.remove((i, j, k))
                    for i, j, k in L:
                        if (i, j, k) in L and (
                            (i == xr and j == xj)
                            or (i == xj and j == xr)
                            or (j == xr and k == xj)
                            or (j == xj and k == xr)
                        ):
                            L.remove((i, j, k))
                    # Delete edge
                    skeleton[xr, xj] = skeleton[xj, xr] = 0
    pdag = skeleton.copy()
    for xi, xj, xk in L:
        pdag[xj, xi] = pdag[xj, xk] = 1
    return pdag, skeleton


def meek(pdag: np.ndarray) -> np.ndarray:
    """
    Orient Meek's rules.

    Args:
        pdag (np.ndarray): The partially directed graph to orient.

    Returns:
        np.ndarray: The completely oriented PDAG.
    """

    def rule1(pdag: np.ndarray) -> bool:
        """
        Rule 1: If i -> j - k and k -/- i, then orient as j -> k.
        """
        i, j = np.where((pdag == -1) & (pdag.T == 1))  # i -> j
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == -1)) & (pdag[i] == 0)  # j - k -/- i
        )
        # Extract actual indices
        i, j, k = i[k[0]], j[k[0]], k[1]
        # Orient as j -> k
        pdag[k, j] = 1
        # return whether graph changed
        return len(i) > 0

    def rule2(pdag: np.ndarray) -> bool:
        """
        Rule 2: If i -> j -> k and i - k, then orient as i -> k.
        """
        i, j = np.where((pdag == -1) & (pdag.T == 1))  # i -> j
        k = np.where(
            ((pdag[j] == -1) & (pdag.T[j] == 1))  # j -> k
            & ((pdag[i] == -1) & (pdag.T[i] == -1))  # i - k
        )
        # Extract actual indices
        i, j, k = i[k[0]], j[k[0]], k[1]
        # Orient as i -> k
        pdag[k, i] = 1
        # return whether graph changed
        return len(i) > 0

    def rule3(pdag: np.ndarray) -> bool:
        """
        Rule 3: If i - j -> l, i - k -> l, j -/- k and i - l, then orient as i -> l.
        """
        i, j = np.where((pdag == -1) & (pdag.T == -1))  # i - j
        k = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1)) & (pdag[j] == 0)  # k - i  #  k -/- j
        )
        # Extract actual indices
        i, j, k = i[k[0]], j[k[0]], k[1]
        mask = j < k
        i, j, k = i[mask], j[mask], k[mask]
        l = np.where(
            ((pdag[i] == -1) & (pdag.T[i] == -1))  # i - l
            & ((pdag[j] == -1) & (pdag.T[j] == 1))  # j -> l
            & ((pdag[k] == -1) & (pdag.T[k] == 1))  # k -> l
        )
        # Extract actual indices
        i, j, k, l = i[l[0]], j[l[0]], k[l[0]], l[1]
        # Orient as i -> l
        pdag[l, i] = 1
        # return whether graph changed
        return len(i) > 0

    pdag = pdag.copy()
    while rule1(pdag) or rule2(pdag) or rule3(pdag):
        continue
    return pdag


def get_poss_anc(g: np.ndarray, targets: list[int]) -> list[int]:
    """
    Get possible ancestors of any target.

    Args:
        g (np.ndarray): The causal graph.
        targets (list[int]): The target nodes.

    Returns:
        list[int]: Possible ancestors of any target.
    """
    g = -g.copy()
    g[g == -1] = 0
    np.fill_diagonal(g, 1)
    g = g.astype(bool)
    # Reachability matrix
    reach = np.linalg.matrix_power(g, g.shape[0] - 1)
    # Nodes that reach at least one target
    poss_anc = np.any(reach[:, targets], axis=1)
    return np.where(poss_anc)[0]


def snap(
    data: np.ndarray,
    ci_test: Callable[[int, int, list[int]], float],
    alpha: float,
    targets: list[int],
    max_order: int = -1,
    **kwargs,
) -> list[int] | dict:
    """
    SNAP algorithm to estimate the CPDAG over the possibly ancestral set of the targets, up to a maximum order of CI tests.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable[[int, int, list[int]], float]): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        targets (list[int]): The target nodes.
        max_order (int): The maximum order of CI tests for SNAP(k). If negative, run SNAP(infinity) until completion (default: -1).
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Adjacency matrix of pruned CPDAG and possible ancestors of targets.
    """
    num_nodes = data.shape[1]
    if max_order < 0:  # Run SNAP(infinity) until completion
        max_order = num_nodes - 1

    # Initialize skeleton with -1 for edge tail, 1 for edge head, and 0 for no edge
    skeleton = np.full((num_nodes, num_nodes), -1, dtype=int)
    np.fill_diagonal(skeleton, 0)
    sep_set = defaultdict(list)

    all_nodes = poss_anc = np.arange(num_nodes)
    for order in range(max_order + 1):
        if np.amax(np.sum(skeleton != 0, axis=1)) <= order:
            break
        # Perform skeleton step
        skeleton = skeleton_step(order, skeleton, ci_test, alpha, sep_set)
        # Orient v-structures
        if order < 2:
            pdag = v_struct_pc(skeleton, sep_set, bidirected=True)
        else:
            pdag, skeleton = v_struct_rfci(skeleton, ci_test, alpha, sep_set)
        # Get possibly ancestral set containing targets
        poss_anc = get_poss_anc(pdag, targets)
        # Prune non-ancestors
        non_anc = np.setdiff1d(all_nodes, poss_anc)
        skeleton[non_anc, :] = skeleton[:, non_anc] = 0

    # If ran until completion
    if max_order == num_nodes - 1:
        # Orient v-structures without bidirected edges
        pdag = v_struct_pc(skeleton, sep_set)
        # Orient Meek rules
        res = meek(pdag)
        # Prune non-ancestors one last time
        poss_anc = get_poss_anc(res, targets)
        non_anc = np.setdiff1d(all_nodes, poss_anc)
        res[non_anc, :] = res[:, non_anc] = 0
    else:
        res = pdag

    # Convert to networkx format
    amat = np.zeros_like(res, dtype=int)
    amat[res == -1] = 1
    return {"amat": amat, "poss_anc": poss_anc}
