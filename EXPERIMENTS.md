# Training experiments

`reproduce.py` is the tested route for recreating all published plots. It uses the supplied reference arrays rather than rerunning expensive stochastic training. `train.py` provides a single configurable implementation for new six-qubit experiments. Its NPZ output schema is deliberately uniform; it does not overwrite the historical dictionaries or pretend to supply their missing metadata.

## Sweep dimensions found in the source

| Figures | Family | Parameters / reference records |
| --- | --- | --- |
| 8 | Strongly entangling layers | Six layers, orders 1–6, three targets, MMD/EMD/TV; stored histories have 20 runs × 800 steps. The TV/SQE mismatch is documented separately. |
| 9 | IQP, matchcircuit, Heisenberg, Haldane | Six qubits, depth 40, five targets, orders 1–6 or matched-size random correlator subsets, ten stored runs per setting. |
| 10 | Batched IQP surrogate | 12 or 20 qubits; bitstring batches 128/256/512; variable constrained correlator count; flip cutoff 2; nominal depth `n+20`; 100 optimization steps. |
| 11a | TN surrogate | Six layers, bond dimensions 4–8; stored curves have ten runs × 200 steps. |
| 11b | IQP PPS | Use the stored summary's available `(k,h)` pairs; five runs × 500 steps; plotting uses the largest `k` and first 200 steps. |
| 12a | TN discrepancy | Bond dimensions 1–8, orders 1–6, five stored runs. |
| 12b–d | IQP discrepancy | Orders 1–6, flip cutoffs 1–8; random IQP, TFIM, and rectangular Haldane targets. Source metadata and legacy MSE conventions are incomplete. |
| 16 | ANOVA training | Degrees 1–6, orders 1–6, six-layer strongly entangling circuit; stored histories have 20 runs × 800 steps. |

The archived curves are the authoritative plot inputs. These dimensions are not a promise that every current notebook default generated the archived run: several notebook defaults had changed, and original seeds were not fully recorded.

## Examples

Run one ANOVA experiment:

```sh
python train.py --target 1d_ising --order 3 --anova-degree 2 --iterations 800 --seed 0 --output runs/anova2_k3_seed0.npz
```

Use EMD or squared Euclidean loss:

```sh
python train.py --loss emd --order 3 --output runs/emd_k3.npz
python train.py --loss sqe --order 3 --output runs/sqe_k3.npz
```

Train a random correlator subset with the same number of correlators as order 3:

```sh
python train.py --ansatz iqp --depth 40 --order 3 --random-subset --output runs/random_subset.npz
```

Train a PPS against a fresh random IQP target (independent deterministic seed offset):

```sh
python train.py --target random_iqp --ansatz iqp --depth 20 --max-flips 4 --order 6 --output runs/pps_random_target.npz
```

For a sweep, vary `--seed` and the explicit parameters and give each run a unique `--output`. The saved `config` field records all settings. Full-support IQP expressions can grow exponentially with depth/flip cutoff. Tensor-network gradients use parameter shift and can be expensive. Small smoke runs establish execution, not convergence or reproduction of the original stochastic outcomes.

`train_batches.py` preserves the source batch objective, including the axis issue explained in `REPRODUCTION_NOTES.md`. Its support masks are little-endian, matching the notebook; full computational-basis probabilities remain big-endian. Do not treat its loss history as a corrected batched probability-distance estimator.
