# The Born Ultimatum

Reproduction code and selected datasets for [The Born Ultimatum: Conditions for Classical Surrogation of Quantum Generative Models with Correlators](https://arxiv.org/abs/2511.01845), arXiv:2511.01845v1.

The repository reproduces **all 17 numbered figures** using the original experiment records or analytical calculations. Figures 1–3 and 13 are the paper's schematic source assets. The remaining figures are computed by one Python module per figure. No original workspace or DMRG installation is needed.

**Read [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) before interpreting the results.** The archived notebooks contain discrepancies between some labels, captions, and calculations. This repository preserves the reference plot values and explicitly records those issues. Replotting the archived results and retraining stochastic models are different operations; fresh training is not claimed to recover the original random runs bit for bit.

## Reproduce the figures

Python 3.12 and a LaTeX installation are required. The figures retain the original serif fonts, LaTeX labels, palettes, and panel organization. TeX Live/MacTeX needs `pdflatex`, `dvipng`, `standalone`, `tikz`, `quantikz2`, and the packages in `diagrams/preamble.tex`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

This writes `outputs/figure_01.pdf` through `outputs/figure_17.pdf`. Select figures or export numerical plots to PNG/PGF:

```sh
python reproduce.py --figures 4,5,9 --format png
python reproduce.py --figures 7,14,15 --format pgf
```

Commands work from any working directory. Schematic figures require PDF output. Reference inputs are never modified by the plotting or training commands.

## Figure map

| Figure | Content | Entry point / inputs |
| --- | --- | --- |
| 1 | Main overview | `diagrams/figure_01.tex` and original sketch |
| 2 | QCBM schematic | `diagrams/figure_02.tex` |
| 3 | Approximation schematic | `diagrams/figure_03.tex` |
| 4 | Hamiltonian ground-state correlators | `born/figures/figure_04.py`; DMRG amplitudes |
| 5 | Gene-activation correlators | `figure_05.py`; `data/n_6_bio.npy` |
| 6 | Matchcircuit variances | `figure_06.py`; analytical |
| 7 | RMPS moments | `figure_07.py`; shared transfer matrices |
| 8 | Loss functions and truncation | `figure_08.py`; `data/diff_losses/` |
| 9 | Algebra/order/subset deployment | `figure_09.py`; `data/G_info*/` |
| 10 | Bitstring/correlator batches | `figure_10.py`; `data/results/` |
| 11 | TN and PPS training curves | `figure_11.py`; `data/TN_data/`, `data/runs_histories/` |
| 12 | Surrogate discrepancy heatmaps | `figure_12.py`; saved MSE records |
| 13 | Ground-state extraction schematic | `diagrams/figure_13.tex` |
| 14 | RMPS subsystem purity panel | `figure_14.py`; analytical |
| 15 | RMPS observable-placement dependence | `figure_15.py`; analytical |
| 16 | ANOVA kernels | `figure_16.py`; loss histories and kernels |
| 17 | Two-qubit parameter dependence | `figure_17.py`; analytical |

Figure numbers follow the paper's reading order, not the inconsistent `fig13`–`fig16` asset filenames in the TeX source. [SOURCE_MAP.json](SOURCE_MAP.json) records the original notebook cells and source hashes.

## Datasets

Only the required DMRG exports are included, without the Julia project or unrelated datasets. [data/README.md](data/README.md) describes their meaning and provenance. Inputs occupy approximately 15 MB. CSV amplitudes are losslessly gzip-compressed; experiment dictionaries are stored as numeric NPZ arrays with JSON structure, requiring no pickle execution or JAX installation for plotting. [data/manifest.json](data/manifest.json) records SHA-256 checksums, original names, and conversion details.

## Fresh training runs

```sh
python -m pip install -r requirements-training.txt
python train.py --target 1d_ising --iterations 200 --output runs/exact.npz
python train.py --ansatz iqp --max-flips 2 --depth 20 --output runs/pps.npz
python train.py --bond-dim 4 --output runs/tn.npz
python train.py --ansatz matchgate --depth 40 --order 3 --output runs/matchgate.npz
python train.py --ansatz iqp --target random_iqp --max-flips 2 --output runs/random_iqp.npz
```

The single training implementation exposes target, circuit family, correlation order, random-subset selection, bond dimension, IQP flip cutoff, loss, ANOVA degree, learning rate, seed, and iteration count. Every result stores its configuration, parameters, loss history, predicted correlators, surrogate probabilities, and exact deployed probabilities. Existing outputs are not overwritten. Raw Fourier truncations may be signed; they are not silently clipped into probability distributions.

For the larger batch experiment:

```sh
python train_batches.py --n 12 --x-batch 128 --z-batch 64 --seed 0 --output runs/batch.npz
```

**This last command preserves a legacy objective with a summation-axis problem**, detailed in the reproduction notes. It is provided to make the source conventions inspectable, not as a validated estimator of the intended batched probability loss.

The original notebook settings, recommended sweep dimensions, and limits of retraining are in [EXPERIMENTS.md](EXPERIMENTS.md). Full historical sweeps have not been rerun; small exact, IQP-surrogate, MPS, and batch smoke runs were checked.

## Checks

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests check dataset hashes, amplitude normalization and bit ordering, Fourier reconstruction, the IQP closed expression against an independent statevector circuit, and the RMPS dynamic program against exhaustive enumeration. Generated outputs, caches, virtual environments, and new runs are ignored by Git.
