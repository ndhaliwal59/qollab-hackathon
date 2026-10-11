"""Classical MIMO detectors and the MMSE + QAOA correction.

Maximum-likelihood detection minimizes ||y - Hx||^2 over x in {-1, +1}.
On the uncertain bits that quadratic expands into an Ising model:

    J_ij = 2 G_ij
    h_i  = 2 (G_{i, F} x_F) - 2 b_i

For complex BPSK, G = Re(H^H H), b = Re(H^H y). New experiments use
signal_rescue.core/hybrid; the QAOA functions below are legacy exploratory APIs.
"""

from __future__ import annotations

import numpy as np

from signal_rescue.qaoa import energy_of, exact_spectrum, random_sample, solve_qaoa

_SYMBOLS: dict[int, np.ndarray] = {}


def _symbols(n: int) -> np.ndarray:
    cached = _SYMBOLS.get(n)
    if cached is None:
        masks = np.arange(1 << n)
        bits = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        cached = 1.0 - 2.0 * bits
        _SYMBOLS[n] = cached
    return cached


def _hard(soft: np.ndarray) -> np.ndarray:
    decision = np.sign(soft).astype(float)
    decision[decision == 0.0] = 1.0
    return decision


def zf_detect(H: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.iscomplexobj(H) or np.iscomplexobj(y):
        H, y = np.concatenate((H.real, H.imag)), np.concatenate((y.real, y.imag))
    soft, *_ = np.linalg.lstsq(H, y, rcond=None)
    return _hard(soft), soft


def mmse_detect(H: np.ndarray, y: np.ndarray, sigma2: float) -> tuple[np.ndarray, np.ndarray]:
    from signal_rescue.core import mmse
    hard, soft, _posterior = mmse(H, y, sigma2)
    return hard, soft


def ml_cost(H: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    residual = y - H @ x
    return float(np.vdot(residual, residual).real)


def exact_ml(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brute-force maximum likelihood. Fine up to about 16 transmit antennas."""
    candidates = _symbols(H.shape[1])
    residual = y[:, None] - H @ candidates.T
    costs = np.sum(np.abs(residual) ** 2, axis=0)
    return candidates[int(np.argmin(costs))].copy()


def local_search(H: np.ndarray, y: np.ndarray, x0: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """Greedy single-bit flips. ``allowed`` restricts which bits may move."""
    x = x0.copy()
    cost = ml_cost(H, y, x)
    free = np.arange(x.shape[0]) if allowed is None else np.asarray(allowed, dtype=int)
    improved = True
    while improved:
        improved = False
        best_i = None
        best_cost = cost
        for i in free:
            x[i] = -x[i]
            trial = ml_cost(H, y, x)
            x[i] = -x[i]
            if trial < best_cost - 1e-9:
                best_cost = trial
                best_i = int(i)
        if best_i is not None:
            x[best_i] = -x[best_i]
            cost = best_cost
            improved = True
    return x


def ising_coefficients(H: np.ndarray, y: np.ndarray, x_hard: np.ndarray, idx: np.ndarray):
    """Ising (h, J) for the free bits in ``idx``. Other bits stay at ``x_hard``."""
    gram = np.real(H.conj().T @ H)
    matched = np.real(H.conj().T @ y)
    idx = np.asarray(idx, dtype=int)
    fixed_mask = np.ones(x_hard.shape[0], dtype=bool)
    fixed_mask[idx] = False
    fixed = np.flatnonzero(fixed_mask)
    coupled = gram[np.ix_(idx, fixed)] @ x_hard[fixed] if fixed.size else 0.0
    h = 2.0 * coupled - 2.0 * matched[idx]
    J = 2.0 * gram[np.ix_(idx, idx)]
    np.fill_diagonal(J, 0.0)
    return h, J


def uncertain_bits(soft: np.ndarray, k: int) -> np.ndarray:
    k = min(k, soft.shape[0])
    return np.argsort(np.abs(soft))[:k]


def reconstruct(x_hard: np.ndarray, idx: np.ndarray, spins: np.ndarray) -> np.ndarray:
    x = x_hard.copy()
    x[np.asarray(idx, dtype=int)] = np.where(spins < 0.0, -1.0, 1.0)
    return x


def _accept(result, h, J, incumbent: np.ndarray) -> np.ndarray:
    """Keep the classical bits unless the sample has a lower Ising cost."""
    if result.energy <= energy_of(incumbent, h, J) + 1e-8:
        return result.spins
    return incumbent


def qaoa_correct(H, y, sigma2, *, k=8, **qaoa_kwargs):
    """MMSE, then QAOA on the k least reliable bits."""
    hard, soft = mmse_detect(H, y, sigma2)
    idx = uncertain_bits(soft, k)
    h, J = ising_coefficients(H, y, hard, idx)
    result = solve_qaoa(h, J, **qaoa_kwargs)
    spins = _accept(result, h, J, hard[idx])
    return reconstruct(hard, idx, spins), result, idx, h, J, hard


def random_correct(H, y, sigma2, *, k=8, shots=64, seed=0):
    """Same uncertain bits as QAOA, but the shots are uniform random."""
    hard, soft = mmse_detect(H, y, sigma2)
    idx = uncertain_bits(soft, k)
    h, J = ising_coefficients(H, y, hard, idx)
    result = random_sample(h, J, shots=shots, seed=seed)
    spins = _accept(result, h, J, hard[idx])
    return reconstruct(hard, idx, spins), result, idx, h, J


def subproblem_optimum(h, J, x_hard, idx) -> np.ndarray:
    _opt, _worst, spins, _masks = exact_spectrum(h, J)
    return reconstruct(x_hard, idx, spins)
