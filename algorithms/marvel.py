import numpy as np
import networkx as nx
import pandas as pd
from typing import Callable

from causallearn.utils.PCUtils import Meek
from rcd.marvel.marvel import Marvel

from utils import FastCausalGraph


class OrientableMarvel(Marvel):
    """Same as original except save separating sets for Section 5.7.1."""

    def __init__(self, ci_test: Callable, mb_fun: Callable):
        super().__init__(ci_test, mb_fun)
        self.x_y_sep_set_mb_dict = dict()  # Section 5.7.1

    # Same as original except save separating sets for Section 5.7.1
    def find_neighborhood(self, var_idx: int):
        """Find the neighborhood of a variable using Lemma 27.

        Args:
            var (int): The variable whose neighborhood we want to find.

        Returns:
            np.ndarray: 1D numpy array containing the variables in the neighborhood.
        """

        var_mk_arr = self.markov_boundary_matrix[var_idx]
        var_mk_idxs = np.flatnonzero(var_mk_arr)

        neighbors_bool_arr = np.copy(var_mk_arr)
        co_parents_bool_arr = np.zeros(len(var_mk_arr), dtype=bool)
        y_sep_set_dict = dict()

        for mb_idx_y in range(len(var_mk_idxs)):
            var_y_idx = var_mk_idxs[mb_idx_y]
            # check if Y is already neighbor of X
            if not self.learned_skeleton.has_edge(var_idx, var_y_idx):
                x_y_sep_set = self.get_sep_set(var_idx, var_y_idx, var_mk_idxs)
                if x_y_sep_set is not None:
                    # var_y is a co-parent of var_idx and thus NOT a neighbor
                    neighbors_bool_arr[var_y_idx] = False
                    co_parents_bool_arr[var_y_idx] = True
                    y_sep_set_dict[var_y_idx] = x_y_sep_set
                    # Save separating set according to Section 5.7.1
                    self.x_y_sep_set_mb_dict[(var_idx, var_y_idx)] = list(x_y_sep_set)

        # remove all variables that are not neighbors
        neighbors_arr = np.flatnonzero(neighbors_bool_arr)
        co_parents_arr = np.flatnonzero(co_parents_bool_arr)
        return neighbors_arr, co_parents_arr, y_sep_set_dict

    # Same as original except save separating sets for Section 5.7.1
    def update_markov_boundary_matrix(self, var_idx: int, var_neighbors: np.ndarray):
        """
        Update the Markov boundary matrix after removing a variable.
        :param var_idx: Index of the variable to remove
        :param var_neighbors: 1D numpy array containing the indices of the neighbors of var_idx
        """
        # for every variable in the markov boundary of var_idx, remove it from the markov boundary and update flag
        for mb_var_idx in np.flatnonzero(
            self.markov_boundary_matrix[var_idx]
        ):  # TODO use indexing instead
            self.markov_boundary_matrix[mb_var_idx, var_idx] = 0
            self.markov_boundary_matrix[var_idx, mb_var_idx] = 0
            self.skip_rem_check_vec[mb_var_idx] = False

        # find nodes whose co-parent status changes
        # we only remove Y from mkvb of Z iff X is their ONLY common child and they are NOT neighbors)
        for ne_idx_y in range(
            len(var_neighbors) - 1
        ):  # -1 because no need to check last variable and also symmetry
            for ne_idx_z in range(ne_idx_y + 1, len(var_neighbors)):
                var_y_idx = var_neighbors[ne_idx_y]
                var_z_idx = var_neighbors[ne_idx_z]
                var_y_name = self.var_names[var_y_idx]
                var_z_name = self.var_names[var_z_idx]

                # determine whether the mkbv of var_y_idx or var_z_idx is smaller, and use the smaller one as cond_set
                var_y_markov_boundary = np.flatnonzero(
                    self.markov_boundary_matrix[var_y_idx]
                )
                var_z_markov_boundary = np.flatnonzero(
                    self.markov_boundary_matrix[var_z_idx]
                )
                if np.sum(self.markov_boundary_matrix[var_y_idx]) < np.sum(
                    self.markov_boundary_matrix[var_z_idx]
                ):
                    cond_set = [
                        self.var_names[idx]
                        for idx in set(var_y_markov_boundary) - {var_z_idx}
                    ]
                else:
                    cond_set = [
                        self.var_names[idx]
                        for idx in set(var_z_markov_boundary) - {var_y_idx}
                    ]
                if self.ci_test(var_y_name, var_z_name, cond_set, self.data):
                    # we know that Y and Z are co-parents and thus NOT neighbors
                    self.markov_boundary_matrix[var_y_idx, var_z_idx] = 0
                    self.markov_boundary_matrix[var_z_idx, var_y_idx] = 0
                    self.skip_rem_check_vec[var_y_idx] = False
                    self.skip_rem_check_vec[var_z_idx] = False
                    # Save separating set according to Section 5.7.1
                    self.x_y_sep_set_mb_dict[(var_y_idx, var_z_idx)] = list(cond_set)


def find_mb_filtered(
    data: pd.DataFrame, ci_test: Callable, ignore: list = []
) -> np.ndarray:
    """
    Computes the Markov boundary matrix for all non-ignored variables.

    Args:
        data: Dataframe where each column is a variable
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a bool.
        ignore: List of variables to ignore
    """

    num_vars = len(data.columns)
    var_name_set = set(data.columns) - set(ignore)
    markov_boundary_matrix = np.zeros((num_vars, num_vars), dtype=bool)

    for i in range(num_vars - 1):  # -1 because no need to check last variable
        if i in ignore:
            continue
        var_name = data.columns[i]
        for j in range(i + 1, num_vars):
            if j in ignore:
                continue
            var_name2 = data.columns[j]
            # check whether var_name and var_name2 are independent of each other given the rest of the variables
            cond_set = list(var_name_set - {var_name, var_name2})
            if not ci_test(var_name, var_name2, cond_set, data):
                markov_boundary_matrix[i, j] = 1
                markov_boundary_matrix[j, i] = 1

    return markov_boundary_matrix


def marvel(
    data: np.ndarray,
    ci_test: Callable,
    alpha: float,
    ignore: list[int] = [],
    **kwargs,
) -> dict:
    """
    MARVEL algorithm

    Args:
        data (np.ndarray): The data matrix.
        ci_test (Callable): CI test taking x, y and a conditioning set, and returns a p-value.
        alpha (float): Significance level.
        ignore (list[int]): Nodes to ignore.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned CPDAG.
    """
    # Create CI test for RCD that returns bools
    rcd_test = (
        lambda x, y, cond_set, *vargs, **kwargs: ci_test(
            x, y, cond_set, *vargs, **kwargs
        )
        >= alpha
    )
    # Learn skeleton
    mb_fun = lambda data: find_mb_filtered(data, rcd_test, ignore)
    marvel = OrientableMarvel(rcd_test, mb_fun)
    skeleton = nx.to_numpy_array(marvel.learn_and_get_skeleton(pd.DataFrame(data)))

    # Orient v-structures based on saved separating sets (Section 5.7.1)
    pdag = skeleton.copy()
    for (x, y), sep_set in marvel.x_y_sep_set_mb_dict.items():
        sep_set = np.array(sep_set)
        common_nb = np.flatnonzero(np.logical_and(skeleton[x] == 1, skeleton[y] == 1))
        v = np.setdiff1d(common_nb, sep_set)  # common neighbors not in sep_set
        pdag[v, x] = pdag[v, y] = 0

    # Orient Meek rules
    pdag = FastCausalGraph.from_amat(pdag)
    pdag.set_ind_test(ci_test)
    cpdag = Meek.meek(pdag)

    cpdag.to_nx_graph()
    amat = nx.to_numpy_array(cpdag.nx_graph).tolist()
    return {"amat": amat}
