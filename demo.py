"""
# SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph

This notebook demonstrates how to use SNAP for causal discovery and its output for causal effect estimation.
We use the [LUCAS0](https://www.causality.inf.ethz.ch/data/LUCAS.html) artificially generated lung cancer dataset
with a known true causal graph, and the PC algorithm as a baseline.
"""

# Import SNAP
from snap import snap

# Import tutorial dependencies
import cProfile
import pstats

from causallearn.utils.cit import CIT
from causallearn.search.ConstraintBased.PC import pc
from dowhy import CausalModel
import networkx as nx
import pandas as pd


# Create a helper function to extract how many CI tests did the algorithms perform.
def count_ci_tests(profiler):
    stats = pstats.Stats(profiler).stats
    cit_call = [
        func for func in stats.keys() if "cit" in func[0] and func[2] == "__call__"
    ][0]
    return stats[cit_call][1]


# Download dataset.
lucas_df = pd.read_csv("http://www.causality.inf.ethz.ch/data/lucas0_train.csv")
lucas_np = lucas_df.to_numpy()
var_labels = {i: name for i, name in enumerate(lucas_df.columns)}
targets = ["Coughing", "Fatigue"]

# Estimate the causal effect according to the true causal graph, which in this case is known.
# True causal graph
lucas_dag = nx.DiGraph()
lucas_dag.add_nodes_from(var_labels.values())
lucas_dag.add_edges_from(
    [
        ("Anxiety", "Smoking"),
        ("Peer_Pressure", "Smoking"),
        ("Smoking", "Yellow_Fingers"),
        ("Smoking", "Lung_cancer"),
        ("Genetics", "Lung_cancer"),
        ("Genetics", "Attention_Disorder"),
        ("Lung_cancer", "Coughing"),
        ("Lung_cancer", "Fatigue"),
        ("Allergy", "Coughing"),
        ("Coughing", "Fatigue"),
        ("Fatigue", "Car_Accident"),
        ("Attention_Disorder", "Car_Accident"),
    ]
)
# Convert to dowhy model
true_model = CausalModel(
    data=lucas_df.copy(),
    treatment=targets[0],  # Coughing
    outcome=targets[1],  # Fatigue
    graph="\n".join(nx.generate_gml(lucas_dag)),
)

# Estimate causal effect according to the true graph
true_identified_estimand = true_model.identify_effect(proceed_when_unidentifiable=True)
oracle_estimate = true_model.estimate_effect(
    true_identified_estimand, method_name="backdoor.propensity_score_weighting"
)

print(
    f"The estimated causal effect of {targets[0]} on {targets[1]} according to the true graph is {oracle_estimate.value}"
)

# As a baseline, we use the PC algorithm. Since the data is binary, we use the G-Square test for discrete variables.
# We use `cProfile` to get how many CI tests PC uses, so we can compare it to SNAP.

# Setup profiler
profiler = cProfile.Profile()

# Run PC algorithm
profiler.enable()
pc_result = pc(data=lucas_np, alpha=0.05, indep_test="chisq", stable=False)
profiler.disable()

# Extract number of CI tests
print(f"PC performed {count_ci_tests(profiler)} number of CI tests")

# Convert to networkx graph
pc_result.to_nx_graph()
nx.relabel_nodes(pc_result.nx_graph, var_labels, copy=False)
# Convert to dowhy model
pc_model = CausalModel(
    data=lucas_df.copy(),
    treatment=targets[0],  # Coughing
    outcome=targets[1],  # Fatigue
    graph="\n".join(nx.generate_gml(pc_result.nx_graph)),
)

# Estimate causal effect according to PC
pc_identified_estimand = pc_model.identify_effect(proceed_when_unidentifiable=True)
pc_estimate = pc_model.estimate_effect(
    pc_identified_estimand, method_name="backdoor.propensity_score_weighting"
)
print(
    f"The estimated causal effect of {targets[0]} on {targets[1]} according to PC is {pc_estimate.value}"
)


# Now we run SNAP($\infty$) with the targets Coughing and Fatigue.
# For flexibility, our implementation of SNAP inputs a callable function (instead of a string) as CI test, thus we create this CI test object explicitly.
# Furthermore, SNAP outputs a dict containing the resulting adjacency matrix.

# Setup CI test, targets and profiler
cit = CIT(data=lucas_df.to_numpy(), indep_test="chisq")
target_inds = [ind for ind, name in var_labels.items() if name in targets]
profiler = cProfile.Profile()

# Run SNAP algorithm
profiler.enable()
snap_result = snap(data=lucas_np, alpha=0.05, ci_test=cit, targets=target_inds)
profiler.disable()

# Extract number of CI tests
print(f"SNAP performed {count_ci_tests(profiler)} number of CI tests")

# Convert to networkx graph
snap_result = nx.DiGraph(snap_result["amat"])
nx.relabel_nodes(snap_result, var_labels, copy=False)
# Convert to dowhy model
snap_model = CausalModel(
    data=lucas_df.copy(),
    treatment=targets[0],  # Coughing
    outcome=targets[1],  # Fatigue
    graph="\n".join(nx.generate_gml(snap_result)),
)

# Estimate causal effect according to SNAP
snap_identified_estimand = snap_model.identify_effect(proceed_when_unidentifiable=True)
snap_estimate = snap_model.estimate_effect(
    snap_identified_estimand, method_name="backdoor.propensity_score_weighting"
)

print(
    f"The estimated causal effect of {targets[0]} on {targets[1]} according to SNAP is {snap_estimate.value}"
)
