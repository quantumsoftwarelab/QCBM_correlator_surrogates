#!/usr/bin/env python3
"""Rerun a specified training experiment; save fresh runs outside reference data."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from born.common import supports, target_probabilities, data_path
from born.models import circuit_correlators, fourier_matrix, iqp_gates, iqp_surrogate, loss_function


def run(config):
    if config["target"] == "random_iqp":
        target_gates = iqp_gates(6, config["depth"], config["seed"] + 100000)
        target_angles = jnp.asarray(
            np.random.default_rng(config["seed"] + 100000).uniform(0, 2 * np.pi, len(target_gates))
        )
        target_circuit = circuit_correlators(6, supports(6), ansatz="iqp", gates=target_gates)
        target = np.asarray(fourier_matrix(6) @ target_circuit(target_angles) / 64)
    else:
        target = target_probabilities(config["target"])
    n = int(np.log2(len(target)))
    seed = config["seed"]
    selected = [s for s in supports(n) if len(s) <= config["order"]]
    if config.get("random_subset", False):
        rng = np.random.default_rng(seed)
        indices = rng.choice(np.arange(1, 2**n), len(selected) - 1, replace=False)
        selected = [()] + [supports(n)[i] for i in sorted(indices)]
    ansatz = config["ansatz"]
    gates = None
    if ansatz == "strongly_entangling":
        shape = (config["layers"], n, 3)
    elif ansatz == "iqp":
        gates = iqp_gates(n, config["depth"], seed)
        shape = (len(gates),)
    else:
        from born.algebras import get_algebra

        algebra = get_algebra(ansatz, n)
        indices = np.random.default_rng(seed).choice(len(algebra), config["depth"])
        gates = [dict(algebra[i]) for i in indices]
        shape = (len(gates),)
    if config.get("max_flips") is not None:
        if ansatz != "iqp":
            raise ValueError("The closed PPS expression applies to IQP circuits")
        evaluate = iqp_surrogate(n, gates, selected, config["max_flips"])
    else:
        evaluate = circuit_correlators(
            n, selected, ansatz, config["layers"], gates, config.get("bond_dim")
        )
    signs = jnp.asarray(fourier_matrix(n, selected))
    kernel = (
        np.load(data_path(f"kernels_6/anova_{config['anova_degree']}.npy"))
        if config.get("anova_degree")
        else None
    )
    objective = loss_function(config["loss"], n, config["sigma"], kernel)

    def loss(angles):
        return objective(signs @ evaluate(angles) / 2**n, jnp.asarray(target))

    angles = 2 * jnp.pi * jax.random.uniform(jax.random.PRNGKey(seed), shape=shape)
    optimizer = optax.adam(config["learning_rate"])
    state = optimizer.init(angles)
    value_grad = jax.value_and_grad(loss)
    # MPS uses parameter-shift callbacks; leave its outer optimization unjitted.
    if config.get("bond_dim") is None:
        value_grad = jax.jit(value_grad)
    history = []
    for _ in range(config["iterations"]):
        value, grad = value_grad(angles)
        updates, state = optimizer.update(grad, state, angles)
        angles = optax.apply_updates(angles, updates)
        history.append(float(value))
    deployed = circuit_correlators(n, supports(n), ansatz, config["layers"], gates)
    exact_correlators = np.asarray(deployed(angles))
    return dict(
        history=np.asarray(history),
        angles=np.asarray(angles),
        correlators=np.asarray(evaluate(angles)),
        probabilities=np.asarray(signs @ evaluate(angles) / 2**n),
        deployed_probabilities=fourier_matrix(n) @ exact_correlators / 2**n,
        target=target,
        config=np.asarray(json.dumps(config)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        choices=["1d_ising", "1d_haldane", "2d_haldane", "1d_heisenberg", "bio", "random_iqp"],
        default="1d_ising",
    )
    p.add_argument(
        "--ansatz",
        choices=["strongly_entangling", "iqp", "matchgate", "heisenberg", "haldane"],
        default="strongly_entangling",
    )
    p.add_argument("--order", type=int, default=6)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--bond-dim", type=int)
    p.add_argument("--max-flips", type=int)
    p.add_argument("--random-subset", action="store_true")
    p.add_argument("--loss", choices=["mmd", "sqe", "emd", "tv"], default="mmd")
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--anova-degree", type=int, choices=range(1, 7))
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = vars(p.parse_args())
    output = args.pop("output")
    if args["iterations"] < 1 or not 1 <= args["order"] <= 6:
        p.error("iterations must be positive and order must be 1..6")
    if args["max_flips"] is not None and args["max_flips"] < 0:
        p.error("max-flips must be nonnegative")
    if output.exists():
        p.error(f"Refusing to overwrite an existing run: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    np.savez_compressed(output, **result)
    print(f"Saved {output}; final pre-update loss {result['history'][-1]:.6g}")


if __name__ == "__main__":
    main()
