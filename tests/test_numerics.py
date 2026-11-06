"""Independent checks for conventions that could silently change the figures."""
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from born.common import correlators, supports, target_probabilities, DATA
from born.tensor_network import observable_moment, order_moments


def test_transfer_dynamic_program_against_exhaustive_strings():
    for n in range(1, 7):
        for bond_dim in (1, 3, 1000):
            expected = np.zeros(n + 1)
            for sequence in product(("id", "pauli"), repeat=n):
                expected[sequence.count("pauli")] += observable_moment(sequence, bond_dim)
            np.testing.assert_allclose(order_moments(n, bond_dim), expected, atol=1e-13)


@pytest.mark.parametrize("name", ["1d_ising", "1d_haldane", "1d_heisenberg", "2d_haldane", "bio"])
def test_target_probability_conventions(name):
    p = target_probabilities(name)
    assert p.shape == (64,)
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(), 1, atol=1e-6)
    actual = correlators(p)
    assert abs(actual[0] - 1) < 1e-6
    expected = []
    for support in supports(6):
        expected.append(
            sum(
                (-1) ** sum(int(format(x, "06b")[q]) for q in support) * px
                for x, px in enumerate(p)
            )
        )
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_data_manifest():
    for item in json.loads((DATA / "manifest.json").read_text()):
        p = DATA / item["path"]
        assert p.stat().st_size == item["bytes"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]


def test_fourier_round_trip():
    from born.models import fourier_matrix

    p = target_probabilities("2d_haldane")
    transform = fourier_matrix(6)
    np.testing.assert_allclose(transform @ correlators(p) / 64, p, atol=1e-14)
    np.testing.assert_allclose(transform.T @ transform, 64 * np.eye(64))


def test_iqp_closed_expression_against_statevector():
    from born.models import iqp_gates, iqp_surrogate, circuit_correlators

    n, depth = 3, 6
    gates = iqp_gates(n, depth, 14)
    selected = supports(n)
    angles = np.random.default_rng(3).uniform(0, 2 * np.pi, len(gates))
    surrogate = iqp_surrogate(n, gates, selected, depth)
    exact = circuit_correlators(n, selected, ansatz="iqp", gates=gates)
    np.testing.assert_allclose(surrogate(angles), exact(angles), atol=1e-12)
