"""Shared circuit, Fourier, loss, and IQP-surrogate implementations."""
from itertools import combinations
import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
from born.common import supports

jax.config.update("jax_enable_x64", True)


def bitstrings(n):
    return (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1


def fourier_matrix(n, selected=None):
    """Rows are binary-ordered outcomes; columns are support-ordered correlators."""
    selected = supports(n) if selected is None else selected
    bits = bitstrings(n)
    return np.stack([1 - 2 * (bits[:, s].sum(axis=1) % 2) for s in selected], axis=1)


def iqp_gates(n, depth, seed):
    """Seeded unique supports, with single-qubit gates first, as in the notebooks."""
    import random

    if not n <= depth <= 2**n - 1:
        raise ValueError("IQP depth must be between n and 2**n-1")
    rng = random.Random(seed)
    gates = [(i,) for i in range(n)]
    seen = set(gates)
    while len(gates) < depth:
        mask = rng.randrange(1, 2**n)
        gate = tuple(i for i in range(n) if mask >> i & 1)
        if gate not in seen:
            gates.append(gate)
            seen.add(gate)
    return gates


def iqp_surrogate(n, gates, selected, max_flips):
    """Compile the closed IQP Pauli-propagation expression once per support.

    Only even-parity flip subsets contribute. Rotation angles use PennyLane's
    exp(-i theta P/2) convention. Real parts remove roundoff imaginary residuals.
    """
    terms = []
    for support in selected:
        active = [j for j, gate in enumerate(gates) if len(set(gate) & set(support)) % 2]
        masks, phases = [], []
        for h in range(min(max_flips, len(active)) + 1):
            for flipped in combinations(range(len(active)), h):
                parity = np.zeros(n, dtype=int)
                for j in flipped:
                    parity[list(gates[active[j]])] ^= 1
                if not parity.any():
                    mask = np.zeros(len(active), dtype=bool)
                    mask[list(flipped)] = True
                    masks.append(mask)
                    phases.append((1j) ** h)
        terms.append(
            (np.array(active, dtype=int), jnp.asarray(np.array(masks)), jnp.asarray(phases))
        )

    def evaluate(angles):
        values = []
        for active, masks, phases in terms:
            theta = angles[active]
            products = jnp.prod(jnp.where(masks, jnp.sin(theta), jnp.cos(theta)), axis=1)
            values.append(jnp.real(jnp.sum(phases * products)))
        return jnp.stack(values)

    return evaluate


def circuit_correlators(
    n, selected, ansatz="strongly_entangling", layers=6, gates=None, bond_dim=None
):
    """One PennyLane implementation for exact and MPS circuit evaluation."""
    if bond_dim is None:
        device = qml.device("default.qubit", wires=n)
        diff_method = "backprop"
    else:
        device = qml.device(
            "default.tensor", wires=n, method="mps", max_bond_dim=bond_dim, contract="auto-mps"
        )
        diff_method = "parameter-shift"
    observables = [qml.prod(*(qml.Z(i) for i in s)) if s else qml.Identity(0) for s in selected]

    @qml.qnode(device, interface="jax", diff_method=diff_method)
    def circuit(angles):
        if ansatz == "strongly_entangling":
            qml.StronglyEntanglingLayers(angles, wires=range(n), ranges=[1] * layers)
        elif ansatz == "iqp":
            for i in range(n):
                qml.Hadamard(i)
            for theta, gate in zip(angles, gates):
                qml.MultiRZ(theta, wires=gate)
            for i in range(n):
                qml.Hadamard(i)
        else:
            for theta, word in zip(angles, gates):
                qml.PauliRot(theta, "".join(word.get(i, "I") for i in range(n)), wires=range(n))
        return tuple(qml.expval(obs) for obs in observables)

    return lambda angles: jnp.stack(circuit(angles))


def loss_function(name, n, sigma=1e-3, kernel=None):
    """Notebook losses with a single implementation and explicit conventions.

    Historical 'hamming' files actually used squared integer-index distance.
    Keep that kernel here for compatibility; EMD likewise uses integer order.
    Truncated Fourier vectors may be signed and are not clipped/renormalized.
    """
    if name == "mmd":
        if kernel is None:
            x = jnp.arange(2**n)
            kernel = jnp.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sigma**2))
        kernel = jnp.asarray(kernel)
        return lambda p, q: (p - q) @ kernel @ (p - q)
    if name == "tv":
        return lambda p, q: 0.5 * jnp.sum(jnp.abs(p - q))
    if name == "sqe":
        return lambda p, q: jnp.sum((p - q) ** 2)
    if name == "emd":
        return lambda p, q: jnp.sum(jnp.abs(jnp.cumsum(p - q)))
    raise ValueError(f"Unknown loss: {name}")
