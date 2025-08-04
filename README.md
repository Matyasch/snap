# SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph

<div align="center">
    
[![paper](https://img.shields.io/badge/paper-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2502.07857)
![website](https://img.shields.io/badge/website-blue?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NDAgNjQwIiBmaWxsPSIjZmZmZmZmIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDcuMC4wIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTQxNS45IDM0NEwyMjUgMzQ0QzIyNy45IDQwOC41IDI0Mi4yIDQ2Ny45IDI2Mi41IDUxMS40QzI3My45IDUzNS45IDI4Ni4yIDU1My4yIDI5Ny42IDU2My44QzMwOC44IDU3NC4zIDMxNi41IDU3NiAzMjAuNSA1NzZDMzI0LjUgNTc2IDMzMi4yIDU3NC4zIDM0My40IDU2My44QzM1NC44IDU1My4yIDM2Ny4xIDUzNS44IDM3OC41IDUxMS40QzM5OC44IDQ2Ny45IDQxMy4xIDQwOC41IDQxNiAzNDR6TTIyNC45IDI5Nkw0MTUuOCAyOTZDNDEzIDIzMS41IDM5OC43IDE3Mi4xIDM3OC40IDEyOC42QzM2NyAxMDQuMiAzNTQuNyA4Ni44IDM0My4zIDc2LjJDMzMyLjEgNjUuNyAzMjQuNCA2NCAzMjAuNCA2NEMzMTYuNCA2NCAzMDguNyA2NS43IDI5Ny41IDc2LjJDMjg2LjEgODYuOCAyNzMuOCAxMDQuMiAyNjIuNCAxMjguNkMyNDIuMSAxNzIuMSAyMjcuOCAyMzEuNSAyMjQuOSAyOTZ6TTE3Ni45IDI5NkMxODAuNCAyMTAuNCAyMDIuNSAxMzAuOSAyMzQuOCA3OC43QzE0Mi43IDExMS4zIDc0LjkgMTk1LjIgNjUuNSAyOTZMMTc2LjkgMjk2ek02NS41IDM0NEM3NC45IDQ0NC44IDE0Mi43IDUyOC43IDIzNC44IDU2MS4zQzIwMi41IDUwOS4xIDE4MC40IDQyOS42IDE3Ni45IDM0NEw2NS41IDM0NHpNNDYzLjkgMzQ0QzQ2MC40IDQyOS42IDQzOC4zIDUwOS4xIDQwNiA1NjEuM0M0OTguMSA1MjguNiA1NjUuOSA0NDQuOCA1NzUuMyAzNDRMNDYzLjkgMzQ0ek01NzUuMyAyOTZDNTY1LjkgMTk1LjIgNDk4LjEgMTExLjMgNDA2IDc4LjdDNDM4LjMgMTMwLjkgNDYwLjQgMjEwLjQgNDYzLjkgMjk2TDU3NS4zIDI5NnoiLz48L3N2Zz4=)

</div>

This is the official code repository for **SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph** (AISTATS 2025) by Mátyás Schubert, Tom Claassen and Sara Magliacane.

> [!NOTE]
> This branch contains all code to reproduce the results presented in the paper. Check out the [main branch](https://github.com/Matyasch/snap/tree/main) for a minimal and portable implementation of SNAP.

Please consult `python3 main.py -h` for parameters. A demo is provided in `demo.ipynb`.

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

## Citation
```bibtex
@inproceedings{schubert2025snap,
    title={{SNAP}: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph},
    author={M{\'a}ty{\'a}s Schubert and Tom Claassen and Sara Magliacane},
    booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
    year={2025},
    url={https://openreview.net/forum?id=0gEjlLdjK9}
}
```
