"""Shared paths, figure style, and computational-basis conventions."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import matplotlib
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREAMBLE = (
    r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}"
    r"\usepackage{dsfont}\usepackage{bm}"
    r"\newcommand{\Var}{\operatorname{\mathds{V}\!ar}}"
)


def data_path(path):
    """Resolve an input relative to the checked-in data directory."""
    path = Path(path)
    if path.suffix == ".pkl":
        path = path.with_suffix(".npz")
    if path.is_absolute():
        raise ValueError(f"Expected a repository-relative dataset path: {path}")
    resolved = (DATA / path).resolve()
    if not resolved.is_relative_to(DATA):
        raise ValueError(f"Dataset path escapes data directory: {path}")
    if not resolved.exists() and resolved.with_name(resolved.name + ".gz").exists():
        return resolved.with_name(resolved.name + ".gz")
    return resolved


def set_style():
    """Keep the original serif/LaTeX, thick-line, pale-spine plot style."""
    sns.set_theme(style="ticks", rc={"figure.facecolor": "white"})
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": PREAMBLE,
            "pgf.preamble": PREAMBLE,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "font.family": "serif",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "lines.linewidth": 3,
            "axes.edgecolor": "lightgray",
            "grid.color": "lightgray",
        }
    )


TARGETS = {
    "1d_ising": "sq_6_1_amplitudes_ising_2d.csv",
    "1d_heisenberg": "sq_6_1_amplitudes_heisenberg_2d.csv",
    "1d_haldane": "sq_6_1_amplitudes_2d.csv",
    "2d_haldane": "sq_2_3_amplitudes_2d.csv",
}


def target_probabilities(name):
    """Return probabilities in binary order, last site varying fastest."""
    if name == "bio":
        return np.load(data_path("n_6_bio.npy"))
    amplitudes = np.loadtxt(data_path(Path("dmrg") / TARGETS[name]), skiprows=1)
    return np.abs(amplitudes) ** 2


def supports(n):
    """Correlator supports ordered by weight, then lexicographically."""
    return [support for k in range(n + 1) for support in combinations(range(n), k)]


def correlators(probabilities):
    """All Z correlators, including identity, in increasing support order."""
    probabilities = np.asarray(probabilities)
    n = int(np.log2(probabilities.size))
    if probabilities.shape != (2**n,):
        raise ValueError("Expected a probability vector of length 2**n")
    states = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return np.array(
        [np.dot(1 - 2 * (states[:, s].sum(axis=1) % 2), probabilities) for s in supports(n)]
    )


def load_results(path):
    """Load numeric reference records without executing pickle payloads."""
    path = Path(path)
    if path.suffix == ".pkl":
        path = path.with_suffix(".npz")
    if not path.is_absolute():
        path = data_path(path)
    with np.load(path, allow_pickle=False) as record:
        tree = json.loads(str(record["structure"]))

        def decode(node):
            if "dict" in node:
                return {key: decode(value) for key, value in node["dict"]}
            return record[node["array"]].copy()

        return decode(tree)
