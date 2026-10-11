"""Small, reusable QAOA circuit constructors.

Keep this module free of training, plotting, and optimizer imports so a Qollab
project can import the circuit without starting the benchmark environment.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _pairs(J: np.ndarray) -> list[tuple[int, int, float]]:
    pairs = []
    for i in range(J.shape[0]):
        for j in range(i + 1, J.shape[0]):
            value = float(J[i, j])
            if value != 0.0:
                pairs.append((i, j, value))
    return pairs


def build_qaoa_circuit(
    h: np.ndarray, J: np.ndarray, p: int
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build standard layered Ising QAOA using Qiskit's abstract gates."""
    h, J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
    if h.ndim != 1 or J.shape != (len(h), len(h)) or not 1 <= p <= 3:
        raise ValueError("Expected h[k], J[k,k], and QAOA depth 1..3")
    gamma, beta = ParameterVector("g", p), ParameterVector("b", p)
    circuit = QuantumCircuit(len(h))
    circuit.h(range(len(h)))
    pairs = _pairs(J)
    for layer in range(p):
        for i, value in enumerate(h):
            if value != 0.0:
                circuit.rz(2.0 * gamma[layer] * float(value), i)
        for i, j, value in pairs:
            circuit.rzz(2.0 * gamma[layer] * value, i, j)
        for i in range(len(h)):
            circuit.rx(2.0 * beta[layer], i)
    return circuit, gamma, beta


def build_ws_qaoa_circuit(
    h: np.ndarray, J: np.ndarray, p: int, theta: np.ndarray
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Build the warm-start circuit retained by the offline research code."""
    h, J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if (h.ndim != 1 or J.shape != (len(h), len(h)) or theta.shape != h.shape
            or not 1 <= p <= 3 or not np.all(np.isfinite(theta))):
        raise ValueError("Expected h[k], J[k,k], finite theta[k], and depth 1..3")
    gamma, beta = ParameterVector("g", p), ParameterVector("b", p)
    circuit = QuantumCircuit(len(h))
    for i, angle in enumerate(theta):
        circuit.ry(float(angle), i)
    pairs = _pairs(J)
    for layer in range(p):
        for i, value in enumerate(h):
            if value != 0.0:
                circuit.rz(2.0 * gamma[layer] * float(value), i)
        for i, j, value in pairs:
            circuit.rzz(2.0 * gamma[layer] * value, i, j)
        for i, angle in enumerate(theta):
            circuit.ry(-float(angle), i)
            circuit.rz(-2.0 * beta[layer], i)
            circuit.ry(float(angle), i)
    return circuit, gamma, beta
