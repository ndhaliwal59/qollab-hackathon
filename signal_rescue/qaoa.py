"""Ising QAOA on a handful of qubits, using only core Qiskit primitives.

The cost is

    E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j,   s_i in {-1, +1}.

``J`` is symmetric and its diagonal is ignored. Only the upper triangle is
read, so ``J_ij`` must be the full coefficient of the pair (not half of it).

Parameters are optimized against the exact ``StatevectorEstimator`` expectation.
The bitstring that is returned comes from a finite ``StatevectorSampler`` shot
budget, not from inspecting every amplitude. With 8 qubits there are only 256
basis states, so a large shot count is just brute force; callers should keep
the budget small and compare against ``random_sample`` at the same budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from signal_rescue.circuits import build_qaoa_circuit, build_ws_qaoa_circuit

_ESTIMATOR: StatevectorEstimator | None = None


def _estimator() -> StatevectorEstimator:
    global _ESTIMATOR
    if _ESTIMATOR is None:
        _ESTIMATOR = StatevectorEstimator()
    return _ESTIMATOR


@dataclass
class QAOAResult:
    spins: np.ndarray
    energy: float
    opt_energy: float
    worst_energy: float
    hit: bool
    prob_optimal: float
    approx_ratio: float
    shots: int
    params: np.ndarray | None
    seconds: float
    n_qubits: int
    p: int
    n_evals: int


def rescale(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Divide coefficients by their max abs value so one angle range fits all."""
    h = np.asarray(h, dtype=float).copy()
    J = np.asarray(J, dtype=float).copy()
    scale = float(np.max(np.abs(h))) if h.size else 0.0
    if J.size:
        scale = max(scale, float(np.max(np.abs(J))))
    if scale < 1e-12:
        scale = 1.0
    return h / scale, J / scale, scale


def _pairs(J: np.ndarray) -> list[tuple[int, int, float]]:
    k = J.shape[0]
    pairs = []
    for i in range(k):
        row = J[i]
        for j in range(i + 1, k):
            val = float(row[j])
            if val != 0.0:
                pairs.append((i, j, val))
    return pairs


def ising_hamiltonian(h: np.ndarray, J: np.ndarray) -> SparsePauliOp:
    """Sparse Pauli form of the Ising cost. Qubit 0 is the rightmost Pauli label."""
    k = len(h)
    labels: list[str] = []
    coeffs: list[float] = []
    for i, hi in enumerate(np.asarray(h, dtype=float)):
        if hi == 0.0:
            continue
        label = ["I"] * k
        label[k - 1 - i] = "Z"
        labels.append("".join(label))
        coeffs.append(float(hi))
    for i, j, val in _pairs(np.asarray(J, dtype=float)):
        label = ["I"] * k
        label[k - 1 - i] = "Z"
        label[k - 1 - j] = "Z"
        labels.append("".join(label))
        coeffs.append(val)
    if not labels:
        return SparsePauliOp.from_list([("I" * k, 0.0)])
    return SparsePauliOp.from_list(list(zip(labels, coeffs))).simplify()


def energy_of(spins: np.ndarray, h: np.ndarray, J: np.ndarray) -> float:
    s = np.asarray(spins, dtype=float)
    e = float(h @ s)
    for i in range(s.shape[0]):
        e += float(s[i] * (J[i, i + 1 :] @ s[i + 1 :]))
    return e


def all_energies(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Energy of every spin configuration. Mask bit i set means spin i is -1."""
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    k = h.shape[0]
    masks = np.arange(1 << k)
    bits = ((masks[:, None] >> np.arange(k)) & 1).astype(float)
    spins = 1.0 - 2.0 * bits
    e = spins @ h
    for i in range(k):
        e = e + spins[:, i] * (spins[:, i + 1 :] @ J[i, i + 1 :])
    return e, spins


def exact_spectrum(h: np.ndarray, J: np.ndarray) -> tuple[float, float, np.ndarray, list[int]]:
    energies, spins = all_energies(h, J)
    opt = float(energies.min())
    worst = float(energies.max())
    opt_masks = [int(i) for i in np.flatnonzero(np.abs(energies - opt) <= 1e-8 * max(1.0, abs(opt)))]
    opt_spins = spins[int(np.argmin(energies))]
    return opt, worst, opt_spins, opt_masks


def mask_to_spins(mask: int, k: int) -> np.ndarray:
    bits = (int(mask) >> np.arange(k)) & 1
    return 1.0 - 2.0 * bits.astype(float)


def _quality(energy: float, opt: float, worst: float) -> float:
    span = worst - opt
    if span < 1e-12:
        return 1.0
    return float(np.clip((worst - energy) / span, 0.0, 1.0))


def _bind(qc: QuantumCircuit, gamma: ParameterVector, beta: ParameterVector, theta: np.ndarray) -> QuantumCircuit:
    p = len(gamma)
    values = {gamma[layer]: float(theta[layer]) for layer in range(p)}
    values.update({beta[layer]: float(theta[p + layer]) for layer in range(p)})
    return qc.assign_parameters(values)


def _expectation(qc: QuantumCircuit, obs: SparsePauliOp, gamma, beta, theta: np.ndarray) -> float:
    bound = _bind(qc, gamma, beta, theta)
    evs = _estimator().run([(bound, obs)]).result()[0].data.evs
    return float(evs)


def _sample_counts(qc: QuantumCircuit, gamma, beta, theta: np.ndarray, shots: int, seed: int) -> dict[int, int]:
    bound = _bind(qc, gamma, beta, theta)
    bound.measure_all()
    sampler = StatevectorSampler(seed=seed)
    bit_array = sampler.run([(bound,)], shots=shots).result()[0].data.meas
    return {int(mask): int(count) for mask, count in bit_array.get_int_counts().items()}


def _best_sample(counts: dict[int, int], h: np.ndarray, J: np.ndarray) -> tuple[int, float]:
    k = len(h)
    best_mask = 0
    best_energy = np.inf
    best_count = -1
    for mask, count in counts.items():
        energy = energy_of(mask_to_spins(mask, k), h, J)
        if energy < best_energy - 1e-9 or (abs(energy - best_energy) <= 1e-9 and count > best_count):
            best_energy = energy
            best_mask = int(mask)
            best_count = count
    return best_mask, float(best_energy)


def _result_from_counts(
    counts: dict[int, int],
    h: np.ndarray,
    J: np.ndarray,
    *,
    shots: int,
    params: np.ndarray | None,
    seconds: float,
    p: int,
    n_evals: int,
) -> QAOAResult:
    opt, worst, _, opt_masks = exact_spectrum(h, J)
    opt_set = set(opt_masks)
    best_mask, best_energy = _best_sample(counts, h, J)
    hit_shots = sum(count for mask, count in counts.items() if mask in opt_set)
    return QAOAResult(
        spins=mask_to_spins(best_mask, len(h)),
        energy=best_energy,
        opt_energy=opt,
        worst_energy=worst,
        hit=best_mask in opt_set,
        prob_optimal=float(hit_shots) / float(shots),
        approx_ratio=_quality(best_energy, opt, worst),
        shots=shots,
        params=None if params is None else np.asarray(params, dtype=float),
        seconds=seconds,
        n_qubits=len(h),
        p=p,
        n_evals=n_evals,
    )


def _initial_points(p: int, n_starts: int, seed: int, x0: np.ndarray | None) -> list[np.ndarray]:
    if x0 is not None:
        return [np.asarray(x0, dtype=float)]
    starts = [np.concatenate([np.full(p, 0.7), np.full(p, 0.5)])]
    rng = np.random.default_rng(seed)
    while len(starts) < max(1, n_starts):
        starts.append(
            np.concatenate(
                [
                    rng.uniform(0.05, np.pi, size=p),
                    rng.uniform(0.05, np.pi, size=p),
                ]
            )
        )
    return starts


def solve_qaoa(
    h,
    J,
    *,
    p: int = 2,
    shots: int = 64,
    seed: int = 0,
    x0: np.ndarray | None = None,
    optimize: bool = True,
    maxiter: int = 40,
    n_starts: int = 3,
) -> QAOAResult:
    """Minimize an Ising cost with QAOA.

    When ``optimize`` is False, ``x0`` is used as-is (parameter transfer).
    Coefficients are rescaled internally; returned energies stay on the
    original scale.
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    if J.shape != (h.shape[0], h.shape[0]):
        raise ValueError(f"J shape {J.shape} does not match h length {h.shape[0]}")
    started = time.perf_counter()
    h_s, J_s, _scale = rescale(h, J)
    qc, gamma, beta = build_qaoa_circuit(h_s, J_s, p)
    n_evals = 0

    if optimize:
        obs = ising_hamiltonian(h_s, J_s)
        best_theta = np.zeros(2 * p)
        best_value = np.inf

        def objective(theta: np.ndarray) -> float:
            # COBYLA reports its last evaluation, not its best, so keep the min here.
            nonlocal n_evals, best_theta, best_value
            n_evals += 1
            value = _expectation(qc, obs, gamma, beta, theta)
            if value < best_value:
                best_value = value
                best_theta = np.array(theta, dtype=float)
            return value

        for theta0 in _initial_points(p, n_starts, seed, x0):
            minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.4})
        theta = best_theta
    else:
        if x0 is None:
            raise ValueError("x0 is required when optimize is False")
        theta = np.asarray(x0, dtype=float)

    counts = _sample_counts(qc, gamma, beta, theta, shots, seed)
    result = _result_from_counts(
        counts,
        h,
        J,
        shots=shots,
        params=theta,
        seconds=time.perf_counter() - started,
        p=p,
        n_evals=n_evals,
    )
    return result


def scaled_expectation(h, J, theta, p: int) -> float:
    """Exact expectation of the rescaled cost. Used to compare transferred parameters."""
    h = np.asarray(h, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    h_s, J_s, _scale = rescale(h, J)
    qc, gamma, beta = build_qaoa_circuit(h_s, J_s, p)
    obs = ising_hamiltonian(h_s, J_s)
    return _expectation(qc, obs, gamma, beta, np.asarray(theta, dtype=float))


def _bias(x_warm, eps: float) -> np.ndarray:
    return np.clip(np.asarray(x_warm, dtype=float).reshape(-1), eps, 1.0 - eps)


def scaled_ws_expectation(h, J, x_warm, theta, p: int, eps: float = 0.25) -> float:
    """Exact expectation of the rescaled cost under the warm-start circuit."""
    h = np.asarray(h, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    h_s, J_s, _scale = rescale(h, J)
    angles = 2.0 * np.arcsin(np.sqrt(_bias(x_warm, eps)))
    qc, gamma, beta = build_ws_qaoa_circuit(h_s, J_s, p, angles)
    obs = ising_hamiltonian(h_s, J_s)
    return _expectation(qc, obs, gamma, beta, np.asarray(theta, dtype=float))


def solve_ws_qaoa(
    h,
    J,
    x_warm,
    *,
    eps: float = 0.25,
    rounds: int = 1,
    p: int = 2,
    shots: int = 64,
    seed: int = 0,
    x0: np.ndarray | None = None,
    optimize: bool = True,
    maxiter: int = 40,
    n_starts: int = 3,
) -> QAOAResult:
    """QAOA from a biased initial state, with an optional iterative update of that bias.

    ``x_warm`` is a binary (or soft) guess. Each round samples ``shots`` bitstrings.
    When ``rounds`` > 1, the next round's bias is the measured marginal of ``x_i = 1``,
    clipped to ``[eps, 1 - eps]``. The returned sample is the last round.
    """
    h = np.asarray(h, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    if J.shape != (h.shape[0], h.shape[0]):
        raise ValueError(f"J shape {J.shape} does not match h length {h.shape[0]}")
    if int(rounds) < 1:
        raise ValueError("rounds must be at least 1")
    warm = np.asarray(x_warm, dtype=float).reshape(-1)
    if warm.shape[0] != h.shape[0]:
        raise ValueError(f"x_warm length {warm.shape[0]} does not match h length {h.shape[0]}")
    started = time.perf_counter()
    h_s, J_s, _scale = rescale(h, J)
    c = _bias(warm, eps)
    n_evals = 0
    params = None if x0 is None else np.asarray(x0, dtype=float)
    if not optimize and params is None:
        raise ValueError("x0 is required when optimize is False")
    counts: dict[int, int] = {}

    for rnd in range(int(rounds)):
        angles = 2.0 * np.arcsin(np.sqrt(c))
        qc, gamma, beta = build_ws_qaoa_circuit(h_s, J_s, p, angles)
        if optimize:
            obs = ising_hamiltonian(h_s, J_s)
            best_theta = np.zeros(2 * p)
            best_value = np.inf

            def objective(theta: np.ndarray) -> float:
                nonlocal n_evals, best_theta, best_value
                n_evals += 1
                value = _expectation(qc, obs, gamma, beta, theta)
                if value < best_value:
                    best_value = value
                    best_theta = np.array(theta, dtype=float)
                return value

            starts = _initial_points(p, n_starts, seed + rnd, params if rnd == 0 else params)
            if rnd > 0 and params is not None:
                starts = [params]
            for theta0 in starts:
                minimize(objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.4})
            params = best_theta
        counts = _sample_counts(qc, gamma, beta, params, shots, seed + rnd * 1009)
        if rnd + 1 < int(rounds):
            total = float(sum(counts.values())) or 1.0
            marginal = np.zeros(h.shape[0])
            for mask, count in counts.items():
                bits = (int(mask) >> np.arange(h.shape[0])) & 1
                marginal += float(count) * bits
            c = np.clip(marginal / total, eps, 1.0 - eps)

    return _result_from_counts(
        counts,
        h,
        J,
        shots=shots,
        params=params,
        seconds=time.perf_counter() - started,
        p=p,
        n_evals=n_evals,
    )


def random_sample(h, J, *, shots: int = 64, seed: int = 0) -> QAOAResult:
    """Pick the lowest-energy sample among ``shots`` uniform random bitstrings."""
    h = np.asarray(h, dtype=float).reshape(-1)
    J = np.asarray(J, dtype=float)
    started = time.perf_counter()
    k = h.shape[0]
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 1 << k, size=shots, endpoint=False)
    counts: dict[int, int] = {}
    for mask in masks:
        key = int(mask)
        counts[key] = counts.get(key, 0) + 1
    return _result_from_counts(
        counts,
        h,
        J,
        shots=shots,
        params=None,
        seconds=time.perf_counter() - started,
        p=0,
        n_evals=0,
    )
