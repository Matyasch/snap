# SNAP: Sequential Non-Ancestor Pruning

<div align="center">

[![paper](https://img.shields.io/badge/paper-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2502.07857)
[![website](https://img.shields.io/badge/website-blue?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NDAgNjQwIiBmaWxsPSIjZmZmZmZmIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDcuMC4wIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTQxNS45IDM0NEwyMjUgMzQ0QzIyNy45IDQwOC41IDI0Mi4yIDQ2Ny45IDI2Mi41IDUxMS40QzI3My45IDUzNS45IDI4Ni4yIDU1My4yIDI5Ny42IDU2My44QzMwOC44IDU3NC4zIDMxNi41IDU3NiAzMjAuNSA1NzZDMzI0LjUgNTc2IDMzMi4yIDU3NC4zIDM0My40IDU2My44QzM1NC44IDU1My4yIDM2Ny4xIDUzNS44IDM3OC41IDUxMS40QzM5OC44IDQ2Ny45IDQxMy4xIDQwOC41IDQxNiAzNDR6TTIyNC45IDI5Nkw0MTUuOCAyOTZDNDEzIDIzMS41IDM5OC43IDE3Mi4xIDM3OC40IDEyOC42QzM2NyAxMDQuMiAzNTQuNyA4Ni44IDM0My4zIDc2LjJDMzMyLjEgNjUuNyAzMjQuNCA2NCAzMjAuNCA2NEMzMTYuNCA2NCAzMDguNyA2NS43IDI5Ny41IDc2LjJDMjg2LjEgODYuOCAyNzMuOCAxMDQuMiAyNjIuNCAxMjguNkMyNDIuMSAxNzIuMSAyMjcuOCAyMzEuNSAyMjQuOSAyOTZ6TTE3Ni45IDI5NkMxODAuNCAyMTAuNCAyMDIuNSAxMzAuOSAyMzQuOCA3OC43QzE0Mi43IDExMS4zIDc0LjkgMTk1LjIgNjUuNSAyOTZMMTc2LjkgMjk2ek02NS41IDM0NEM3NC45IDQ0NC44IDE0Mi43IDUyOC43IDIzNC44IDU2MS4zQzIwMi41IDUwOS4xIDE4MC40IDQyOS42IDE3Ni45IDM0NEw2NS41IDM0NHpNNDYzLjkgMzQ0QzQ2MC40IDQyOS42IDQzOC4zIDUwOS4xIDQwNiA1NjEuM0M0OTguMSA1MjguNiA1NjUuOSA0NDQuOCA1NzUuMyAzNDRMNDYzLjkgMzQ0ek01NzUuMyAyOTZDNTY1LjkgMTk1LjIgNDk4LjEgMTExLjMgNDA2IDc4LjdDNDM4LjMgMTMwLjkgNDYwLjQgMjEwLjQgNDYzLjkgMjk2TDU3NS4zIDI5NnoiLz48L3N2Zz4=)](https://matyasch.github.io/snap/)
[![demo](https://img.shields.io/badge/demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1SR3u3GPVp_EXlxZpJTDrn87rBUKKR7wA#scrollTo=Jpzktsfv_K2O)


</div>

This is the official code repository for **SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph** (AISTATS 2025) by Mátyás Schubert, Tom Claassen and Sara Magliacane.

> [!NOTE]
> This branch contains a minimal and portable implementation of SNAP. Check out the [aistats2025 branch](https://github.com/Matyasch/snap/tree/aistats2025) to reproduce the results presented in the paper.

The SNAP algorithm is implemented in [`snap.py`](snap.py). A simple demo is provided in [`demo.py`](demo.py) and in [Google colab](https://colab.research.google.com/drive/1SR3u3GPVp_EXlxZpJTDrn87rBUKKR7wA#scrollTo=t_MOscYctYG4). All dependencies are listed in [`requirements.txt`](requirements.txt)


## Citation
```bibtex
@InProceedings{pmlr-v258-schubert25a,
  title = {SNAP: Sequential Non-Ancestor Pruning for Targeted Causal Effect Estimation With an Unknown Graph},
  author = {Schubert, M{\'a}ty{\'a}s and Claassen, Tom and Magliacane, Sara},
  booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
  pages = {3340--3348},
  year = {2025},
  editor = {Li, Yingzhen and Mandt, Stephan and Agrawal, Shipra and Khan, Emtiyaz},
  volume = {258},
  series = {Proceedings of Machine Learning Research},
  month = {03--05 May},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v258/main/assets/schubert25a/schubert25a.pdf},
  url = {https://proceedings.mlr.press/v258/schubert25a.html},
}
```
