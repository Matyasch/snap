from contextlib import chdir
import os
import sys

import numpy as np
import pandas as pd

from utils import CountingTest

sys.path.insert(1, os.getcwd() + "/py-tetrad/pytetrad")
with chdir("py-tetrad/pytetrad"):
    import tools.TetradSearch as ts


def fges(
    data: np.ndarray,
    ci_test: CountingTest,
    ignore: list[int] = [],
    **kwargs,
) -> dict:
    """
    FGES algorithm.

    Args:
        data (np.ndarray): The data matrix.
        ci_test (CountingTest): CI test to determine appropriate score function.
        ignore (list[int]): Nodes to ignore.
        **kwargs: Additional arguments are ignored.

    Returns:
        dict: Learned CPDAG.
    """
    # Prepare data
    nodes = data.shape[1]
    if ci_test.method == "chisq":
        data = data.astype(int)
    data = pd.DataFrame(data)
    data.drop(columns=ignore, inplace=True)
    remaining = list(data.columns)
    # Run FGES
    search = ts.TetradSearch(data)
    search.set_verbose(False)
    if ci_test.method == "fisherz":
        search.use_sem_bic()
    elif ci_test.method == "chisq":
        search.use_bdeu()
    search.run_fges()
    # Convert result
    tetrad_amat = search.get_graph_to_matrix().to_numpy()
    tetrad_amat = tetrad_amat
    pruned_amat = np.zeros_like(tetrad_amat)
    pruned_amat[tetrad_amat == 2] = 1
    pruned_amat[np.logical_and(tetrad_amat == 3, tetrad_amat.T == 3)] = 1
    # Add ignored nodes back
    full_amat = np.zeros((nodes, nodes))
    full_amat[np.ix_(remaining, remaining)] = pruned_amat
    return {"amat": full_amat.tolist()}
