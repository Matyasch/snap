from argparse import ArgumentParser
from concurrent.futures import as_completed, ProcessPoolExecutor
from functools import partial
import sys
from time import process_time
from typing import Callable

import numpy as np
from tqdm import tqdm

from algorithms import ALGORITHMS, snap
import utils


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--exp", type=int, default=100, help="Number of experiments.")
    parser.add_argument("--file", type=str, help="Data file path.")
    parser.add_argument("--nodes", type=int, help="Number of nodes.")
    parser.add_argument("--targets", type=int, default=4, help="Number of targets.")
    parser.add_argument(
        "--exp-degree", type=float, default=3.0, help="Expected degree."
    )
    parser.add_argument("--max-degree", type=int, default=10, help="Maximum degree.")
    parser.add_argument(
        "--connected", action="store_true", help="Generate connected graphs."
    )
    parser.add_argument(
        "--identifiable",
        action="store_true",
        help="Ensure that all targets are identifiable.",
    )
    parser.add_argument(
        "--min-adj-size",
        type=int,
        default=0,
        help="Minimum adjustment set size to identify (non-zero) causal effects.",
    )
    parser.add_argument(
        "--citest",
        type=str,
        help="Conditional independence test.",
    )
    parser.add_argument("--samples", type=int, default=0, help="Number of samples.")
    parser.add_argument(
        "--discrete", action="store_true", help="Generate discrete data."
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS.keys(),
        required=True,
        help="Causal discovery algorithm.",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=-1,
        help="Max order for filtering non-ancestors.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed.")
    args = parser.parse_args(sys.argv[1:])
    return args


def run_algorithm(
    id: str,
    data: np.ndarray,
    targets: list[int],
    algorithm: Callable,
    filter_order: int,
    ci_test: str,
    alpha: float,
    **kwargs
) -> dict:
    """
    Run the given causal discovery algorithm on the given data.

    Args:
        id (str): Experiment ID.
        data (np.ndarray): The data matrix.
        targets (list[int]): The target nodes.
        algorithm (Callable): The causal discovery algorithm to run.
        filter_order (int): The maximum order of CI tests for pre-filtering.
        ci_test (str): The conditional independence test to use.
        alpha (float): Significance level.
        **kwargs: Additional arguments for the algorithm.

    Returns:
        dict: Results of the experiment.
    """
    ci_tester = utils.CountingTest(data, ci_test, **kwargs)
    start = process_time()
    # Pre-filter non-ancestors with SNAP
    if filter_order >= 0:
        ignore = snap(
            data=data,
            ci_test=ci_tester,
            alpha=alpha,
            targets=targets,
            max_order=filter_order,
            filter_mode=True,
            oracle=(ci_test == "d_separation"),
            **kwargs,
        )["non_anc"]
    else:
        ignore = []

    filter_tests = ci_tester.get_tests_per_order()
    # Run causal discovery algorithm
    result = algorithm(
        data=data,
        ci_test=ci_tester,
        alpha=alpha,
        targets=targets,
        ignore=ignore,
        oracle=(ci_test == "d_separation"),
        **kwargs,
    )
    elapsed = process_time() - start

    tests = ci_tester.get_tests_per_order() - filter_tests
    result["id"] = id
    result["targets"] = targets
    result["time"] = elapsed
    result["tests"] = tests.tolist()
    result["filter_tests"] = filter_tests.tolist()
    return result


def generate_data_and_run_algorithm(
    algorithm: Callable,
    nodes: int,
    seed: int = 0,
    file: str | None = None,
    exp_degree: float = 3.0,
    max_degree: int = 10,
    targets: int = 4,
    connected: bool = True,
    identifiable: bool = False,
    min_adj_size: int = 0,
    samples: int = 1000,
    discrete: bool = False,
    filter_order: int = -1,
    ci_test: str = None,
    alpha: float = 0.05,
) -> dict:
    """
    Run a single experiment by generating data and running the causal discovery algorithm.

    Args:
        nodes (int): Number of nodes.
        seed (int): Local seed.
        exp (int): Number of experiments.
        file (str): nblearn object file path for data generation.
        exp_degree (float): Expected degree.
        max_degree (int): Maximum degree.
        targets (int): Number of targets.
        connected (bool): Whether to generate connected graphs.
        identifiable (bool):  Whether to ensure that all targets are identifiable.
        min_adj_size (int): Minimum adjustment set size to consider causal effect identifiable.
        samples (int): Number of samples.
        discrete (bool): Whether to generate discrete data.
        algorithm (Callable): The causal discovery algorithm to run.
        filter_order (int): The maximum order of CI tests for pre-filtering.
        ci_test (str): The conditional independence test to use.
        alpha (float): Significance level.

    Returns:
        list[dict]: Results of the experiments.
    """
    data = utils.generate_data(
        seed=seed,
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
    result = run_algorithm(
        algorithm=algorithm,
        filter_order=filter_order,
        ci_test=ci_test,
        alpha=alpha,
        **data,
    )
    return result


def run_experiments(
    algorithm: Callable,
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
    filter_order: int = -1,
    ci_test: str = None,
    alpha: float = 0.05,
) -> list[dict]:
    """
    Run multiple experiments in parallel with the given arguments.

    Args:
        nodes (int): Number of nodes.
        seed (int): Global seed.
        exp (int): Number of experiments.
        file (str): nblearn object file path for data generation.
        exp_degree (float): Expected degree.
        max_degree (int): Maximum degree.
        targets (int): Number of targets.
        connected (bool): Whether to generate connected graphs.
        identifiable (bool):  Whether to ensure that all targets are identifiable.
        min_adj_size (int): Minimum adjustment set size to consider causal effect identifiable.
        samples (int): Number of samples.
        discrete (bool): Whether to generate discrete data.
        algorithm (Callable): The causal discovery algorithm to run.
        filter_order (int): The maximum order of CI tests for pre-filtering.
        ci_test (str): The conditional independence test to use.
        alpha (float): Significance level.

    Returns:
        list[dict]: Results of the experiments.
    """
    # Fisher-Z and KCI implementations were faster without parallelism
    workers = 1 if ci_test in ["fisherz", "kci"] else None

    procs = []
    with ProcessPoolExecutor(workers) as exec:
        for s in range(exp):
            process = exec.submit(
                generate_data_and_run_algorithm,
                # Parameters for data generation
                seed=seed + s,
                file=file,
                nodes=nodes,
                exp_degree=exp_degree,
                max_degree=max_degree,
                targets=targets,
                connected=connected,
                identifiable=identifiable,
                min_adj_size=min_adj_size,
                samples=samples,
                discrete=discrete,
                # Parameters for causal discovery algorithm
                algorithm=algorithm,
                filter_order=filter_order,
                ci_test=ci_test,
                alpha=alpha,
            )
            procs.append(process)

        results = []
        for p in tqdm(as_completed(procs), total=len(procs)):
            results.append(p.result())
    return results


if __name__ == "__main__":
    args = parse_args()

    run_experiments(
        algorithm=ALGORITHMS[args.algorithm],
        seed=args.seed,
        exp=args.exp,
        file=args.file,
        nodes=args.nodes,
        exp_degree=args.exp_degree,
        max_degree=args.max_degree,
        targets=args.targets,
        connected=args.connected,
        identifiable=args.identifiable,
        min_adj_size=args.min_adj_size,
        samples=args.samples,
        discrete=args.discrete,
        filter_order=args.filter_order,
        ci_test=args.citest,
        alpha=args.alpha,
    )
