"""The notebooks' two-by-two RMPS transfer matrices, shared across figures."""
import numpy as np


def observable_matrix(kind, bond_dim, local_dim=2):
    """Second-moment transfer matrix; retain the published boundary convention."""
    traces = {"id": (local_dim, local_dim), "proj": (1, 1), "pauli": (0, local_dim)}
    tr, tr2 = traces[kind]
    d, D = local_dim, bond_dim
    return np.array(
        [
            [d * D**2 * tr**2 - tr2, d * D * tr**2 - D * tr2],
            [-D * tr**2 + d * D * tr2, -(tr**2) + d * D**2 * tr2],
        ]
    ) / (d * (d**2 * D**2 - 1))


def observable_moment(sequence, bond_dim):
    product = np.eye(2)
    for kind in sequence:
        product = product @ observable_matrix(kind, bond_dim)
    return product.sum()


def order_moments(n, bond_dim):
    """Sum over every Pauli/identity string at each weight, without 2**n storage.

    Matrix-valued polynomial multiplication preserves the order of the original
    noncommuting factors. Coefficient k sums all strings with k Pauli factors.
    """
    coeffs = np.zeros((n + 1, 2, 2))
    coeffs[0] = np.eye(2)
    identity = observable_matrix("id", bond_dim)
    pauli = observable_matrix("pauli", bond_dim)
    for length in range(n):
        updated = np.zeros_like(coeffs)
        updated[: length + 1] += coeffs[: length + 1] @ identity
        updated[1 : length + 2] += coeffs[: length + 1] @ pauli
        coeffs = updated
    return coeffs.sum(axis=(1, 2))


def purity_moment(sequence, bond_dim, local_dim=2):
    d, D = local_dim, bond_dim
    identity = np.array(
        [[1.0, D * (d * d - 1) / (d * d * D * D - 1)], [0.0, d * (D * D - 1) / (d * d * D * D - 1)]]
    )
    product = np.eye(2)
    for kind in sequence:
        product = product @ (identity if kind == "i" else identity[::-1, ::-1])
    return product.sum()
