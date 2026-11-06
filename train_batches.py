#!/usr/bin/env python3
"""Reproduce the archived batch experiment's conventions (see REPRODUCTION_NOTES)."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
from born.common import data_path
from born.models import iqp_surrogate, iqp_gates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, choices=[6, 12, 20], default=12)
    p.add_argument("--x-batch", type=int, default=128)
    p.add_argument("--z-batch", type=int, default=64)
    p.add_argument("--max-flips", type=int, default=2)
    p.add_argument("--depth", type=int)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}")
    if not 0 < args.z_batch < 2**args.n or args.x_batch < 1 or args.iterations < 1:
        p.error("Batch sizes and iteration count must be positive; z-batch < 2**n")
    if args.max_flips < 0:
        p.error("max-flips must be nonnegative")
    n = args.n
    depth = args.depth or n + 20
    if not n <= depth < 2**n:
        p.error("depth must be between n and 2**n-1")
    rng = random.Random(args.seed)
    gates = iqp_gates(n, depth, args.seed)
    masks = rng.sample(range(1, 2**n), args.z_batch)
    selected = [tuple(i for i in range(n) if mask >> i & 1) for mask in masks]
    target = (
        np.abs(np.loadtxt(data_path(f"dmrg/sq_{n}_1_amplitudes_ising_2d.csv"), skiprows=1)) ** 2
    )
    # Preserve the notebook's initialization and little-endian support indexing.
    angles = jnp.asarray([np.sqrt(np.prod(target[list(gate)])) for gate in gates])
    evaluate = iqp_surrogate(n, gates, selected, args.max_flips)
    observed_target = jnp.asarray(target[masks])

    def loss(theta, signs):
        # Historical code sums over sampled outcomes for each correlator.
        # This is NOT a probability evaluated at each sampled outcome.
        estimate = (jnp.sum(signs * evaluate(theta)[:, None], axis=1) + 1) / 2**n
        return jnp.sum((estimate - observed_target) ** 2)

    value_grad = jax.jit(jax.value_and_grad(loss))
    opt = optax.adam(0.1)
    state = opt.init(angles)
    history = []
    sample_rng = np.random.default_rng(args.seed)
    for _ in range(args.iterations):
        bits = sample_rng.integers(0, 2, (args.x_batch, n))
        signs = jnp.asarray(np.stack([1 - 2 * (bits[:, s].sum(axis=1) % 2) for s in selected]))
        value, grad = value_grad(angles, signs)
        updates, state = opt.update(grad, state, angles)
        angles = optax.apply_updates(angles, updates)
        history.append(float(value))
    device = qml.device("default.qubit", wires=n)

    @qml.qnode(device)
    def deployed(theta):
        for i in range(n):
            qml.Hadamard(i)
        for angle, gate in zip(theta, gates):
            qml.MultiRZ(angle, wires=gate)
        for i in range(n):
            qml.Hadamard(i)
        return qml.probs(wires=range(n))

    probabilities = np.asarray(deployed(np.asarray(angles)))
    kl = np.sum(probabilities * np.log((probabilities + 1e-15) / (target + 1e-15)))
    config = vars(args).copy()
    config["output"] = str(args.output)
    gate_masks = np.array([sum(1 << i for i in gate) for gate in gates])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        angles=np.asarray(angles),
        history=history,
        gate_masks=gate_masks,
        support_masks=masks,
        deployed_probabilities=probabilities,
        deployed_kl=kl,
        config=np.asarray(json.dumps(config)),
    )
    print(f"Saved {args.output}; deployed KL {kl:.6g}")


if __name__ == "__main__":
    main()
