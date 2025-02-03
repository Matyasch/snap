# SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph

This is the official code repository for **SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph** (AISTATS 2025) by Mátyás Schubert, Tom Claassen and Sara Magliacane.

Please consult `python3 main.py -h` for parameters. A demo is provided in `demo.ipynb`. The SNAP algorithm is implemented in [`algorithms/snap.py`](algorithms/snap.py).

## Requirements
Python dependencies can be installed with `pip3 install -r requirements.txt`

R dependencies can be installed as follows
```R
install.packages("BiocManager")
BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
install.packages(c("pcalg", "igraph", "expm", "bnlearn", "dagitty"))
```

Install py-tetrad such that the repository is cloned into **this** directory, following the [official github page](https://github.com/cmu-phil/py-tetrad?tab=readme-ov-file#install).

The ldecc_github directory contains code adapted from https://github.com/acmi-lab/local-causal-discovery.