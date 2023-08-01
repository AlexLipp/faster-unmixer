import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, TypeVar, Optional, DefaultDict

from random_networks_test import (
    generate_r_ary_sample_network,
    draw_random_log_uniform,
    conc_list_to_dict,
)
import funmixer
import matplotlib.pyplot as plt
import numpy as np
import tqdm

T = TypeVar("T")

# 100 networks up to 500 nodes in size takes about 12 minutes to run on my machine

# Set the range to explore upstream concentration values over
MINIMUM_CONC = 1
MAXIMUM_CONC = 1e2

# Set the range to explore sub-basin area values over
MINIMUM_AREA = 1
MAXIMUM_AREA = 1e2

# Set branching factor
BRANCHING_FACTOR = 3

# Set maximum number of nodes to try
MAXIMUM_NETWORK_SIZE = 500

# Set number of networks to test
NUMBER_OF_NETWORKS = 100


@dataclass
class BenchmarkResults:
    first_solves: DefaultDict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    subsequent_solves: DefaultDict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    solver_times: DefaultDict[int, List[float]] = field(default_factory=lambda: defaultdict(list))


def bavg(x: DefaultDict[int, List[float]]) -> List[float]:
    return [float(np.mean(x[n])) for n in sorted(x.keys())]


def bstd(x: DefaultDict[int, List[float]]) -> List[float]:
    return [float(np.std(x[n])) for n in sorted(x.keys())]


def plot_first_second(
    network_sizes: List[int], b: BenchmarkResults, name: str, c1: str, c2: str
) -> None:
    plt.plot(
        network_sizes,
        bavg(b.first_solves),
        "o-",
        c=c1,
        label=f"{name} (1$^{{st}}$ solve)",
        markersize=4,
    )
    plt.errorbar(
        network_sizes,
        bavg(b.first_solves),
        yerr=bstd(b.first_solves),
    )
    plt.plot(
        network_sizes,
        bavg(b.subsequent_solves),
        "o-",
        c=c2,
        # Make points smaller
        markersize=4,
        label=f"{name} (2$^{{nd}}$ solve)",
    )
    plt.errorbar(
        network_sizes,
        bavg(b.subsequent_solves),
        yerr=bstd(b.subsequent_solves),
    )


def plot_solver_time(network_sizes: List[int], b: BenchmarkResults, name: str, c: str) -> None:
    plt.plot(
        network_sizes,
        bavg(b.solver_times),
        "o-",
        c=c,
        label=f"{name}",
        markersize=4,
    )
    plt.errorbar(
        network_sizes,
        bavg(b.solver_times),
        yerr=bstd(b.solver_times),
    )


def none_throws(x: Optional[T]) -> T:
    assert x is not None
    return x


def run_benchmark(solver: str, sizes: List[int], repeats: int = 10) -> BenchmarkResults:
    ret = BenchmarkResults()

    for n in tqdm.tqdm(sizes):
        for _ in range(repeats):
            areas = lambda: draw_random_log_uniform(MINIMUM_AREA, MAXIMUM_AREA)
            concentrations = lambda: draw_random_log_uniform(MINIMUM_CONC, MAXIMUM_CONC)
            network = generate_r_ary_sample_network(
                N=n, branching_factor=BRANCHING_FACTOR, areas=areas
            )
            upstream = conc_list_to_dict(network, concentrations)
            downstream = funmixer.forward_model(
                sample_network=network, upstream_concentrations=upstream
            )
            problem = funmixer.SampleNetworkUnmixer(
                sample_network=network, use_regularization=False
            )
            solution = problem.solve(downstream, solver=solver)
            ret.first_solves[n].append(solution.total_time)
            ret.solver_times[n].append(none_throws(solution.solve_time))
            solution = problem.solve(downstream, solver=solver)
            ret.subsequent_solves[n].append(solution.total_time)

    return ret


def main() -> None:
    # Generate list of network sizes
    network_sizes = np.unique(
        np.rint(
            np.logspace(np.log10(2), np.log10(MAXIMUM_NETWORK_SIZE), NUMBER_OF_NETWORKS)
        ).astype(int)
    )

    # NOTE: Select a slice here for doing dev work on the benchmarks
    network_sizes = network_sizes[:]

    print("#" * 80)
    print(
        f"Running benchmark for {network_sizes.size} R-ary networks with branching factor {BRANCHING_FACTOR}, up to {MAXIMUM_NETWORK_SIZE} nodes."
    )
    print(
        f"Node concentrations and areas randomly varied between: {MINIMUM_CONC} and {MAXIMUM_CONC}, {MINIMUM_AREA} and {MAXIMUM_AREA}, respectively."
    )
    print("#" * 80)
    start = time.time()

    print("Testing ECOS solver...")
    ecos_bench = run_benchmark("ecos", network_sizes)

    print("Testing GUROBI solver...")
    gurobi_bench: Optional[BenchmarkResults] = None
    try:
        gurobi_bench = run_benchmark("gurobi", network_sizes)
    except Exception as err:
        print(f"Could not benchmark Gurobi. Error: {err}")

    print("Testing SCS solver...")
    scs_bench = run_benchmark("scs", network_sizes)

    end = time.time()
    print(f"Benchmarking took {end - start} seconds.")
    print("Benchmark complete.")
    print("#" * 80)
    # Plot results of total runtime
    # Plot solve time against number of nodes
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plot_first_second(network_sizes, ecos_bench, "ECOS", "#1f78b4", "#a6cee3")
    plot_first_second(network_sizes, scs_bench, "SCS", "#e31a1c", "#fb9a99")
    if gurobi_bench:
        plot_first_second(network_sizes, gurobi_bench, "GUROBI", "#33a02c", "#b2df8a")

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Total runtime")
    plt.ylabel("Solve time (s)")
    plt.legend()
    plt.grid(True, which="both")
    plt.subplot(2, 1, 2)

    # Plot results of just the CVXPY solve time
    plot_solver_time(network_sizes, ecos_bench, "ECOS", "#1f78b4")
    plot_solver_time(network_sizes, scs_bench, "SCS", "#e31a1c")
    if gurobi_bench:
        plot_solver_time(network_sizes, gurobi_bench, "GUROBI", "#33a02c")

    plt.xscale("log")
    plt.title("Optimizer time")
    plt.yscale("log")
    plt.xlabel("Number of nodes")
    plt.ylabel("Solve time (s)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("runtime_benchmark.png", dpi=400, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
