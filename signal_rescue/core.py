"""Classical BPSK inference and bounded refinement; no quantum dependencies."""

from __future__ import annotations
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=16)
def spin_table(n):
    if not 1 <= n <= 16:
        raise ValueError("Spin table supports 1..16 bits")
    table = 1. - 2. * ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1)
    table.setflags(write=False)
    return table


def validate(H, y, sigma2):
    H, y = np.asarray(H), np.asarray(y)
    if (H.ndim != 2 or y.shape != (H.shape[0],) or not H.shape[0] or not H.shape[1]
            or not np.all(np.isfinite(H)) or not np.all(np.isfinite(y))
            or not np.isfinite(sigma2) or sigma2 < 0):
        raise ValueError("Expected finite H[rx,tx], y[rx], and nonnegative noise variance")
    return H, y


def quadratic(H, y):
    return np.real(H.conj().T @ H), np.real(H.conj().T @ y)


def cost(H, y, x):
    residual = y - H @ x
    return float(np.vdot(residual, residual).real)


def mmse(H, y, sigma2):
    H, y = validate(H, y, sigma2)
    G, b = quadratic(H, y)
    variance = sigma2 / 2 if np.iscomplexobj(H) or np.iscomplexobj(y) else sigma2
    reg = max(float(variance), 1e-12)
    inverse = np.linalg.inv(G + reg * np.eye(G.shape[0]))
    soft = inverse @ b
    posterior = np.maximum(reg * np.diag(inverse), 1e-12)
    return np.where(soft < 0, -1., 1.), soft, posterior


def ordered_sic(H, y, sigma2):
    H, y = validate(H, y, sigma2)
    remaining = list(range(H.shape[1]))
    residual = y.copy()
    x = np.ones(H.shape[1])
    while remaining:
        hard, _, posterior = mmse(H[:, remaining], residual, sigma2)
        local = int(np.argmin(posterior))
        original = remaining.pop(local)
        x[original] = hard[local]
        residual -= H[:, original] * x[original]
    return x


@lru_cache(maxsize=16)
def flip_sets(n, pair=True):
    sets = [(i,) for i in range(n)]
    if pair:
        sets += [(i, j) for i in range(n) for j in range(i + 1, n)]
    mask = np.zeros((len(sets), n))
    for row, indices in enumerate(sets):
        mask[row, list(indices)] = 1
    mask.setflags(write=False)
    return mask


def refine(G, b, candidates, *, steps=8, pair=True):
    """Identical bounded single/pair search for every sampler's candidates."""
    x = np.atleast_2d(candidates).astype(float).copy()
    if steps < 0 or np.any(np.abs(x) != 1):
        raise ValueError("Refinement requires spins and a nonnegative step budget")
    n = x.shape[1]
    masks = flip_sets(n, pair)
    left, right = np.triu_indices(n, 1)
    moves = 0
    for _ in range(steps):
        gradient = x @ G - b
        singles = -4 * x * gradient + 4 * np.diag(G)
        changes = singles
        if pair:
            pairs = singles[:, left] + singles[:, right] + 8 * x[:, left] * x[:, right] * G[left, right]
            changes = np.concatenate((singles, pairs), axis=1)
        best = np.argmin(changes, axis=1)
        improve = changes[np.arange(len(x)), best] < -1e-10
        if not np.any(improve):
            break
        x[improve] *= 1 - 2 * masks[best[improve]]
        moves += int(np.sum(improve))
    return x, moves


def ising(G, b, incumbent, idx):
    idx = np.asarray(idx, dtype=int)
    if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= len(incumbent)):
        raise ValueError("Free-bit indices must be unique and in range")
    fixed = np.setdiff1d(np.arange(len(incumbent)), idx)
    h = 2 * G[np.ix_(idx, fixed)] @ incumbent[fixed] - 2 * b[idx]
    J = 2 * G[np.ix_(idx, idx)].copy()
    np.fill_diagonal(J, 0)
    return h, J


def energies(spins, h, J):
    spins = np.atleast_2d(spins)
    return spins @ h + np.sum((spins @ np.triu(J, 1)) * spins, axis=1)


def select_block(soft, posterior, G, k, *, coupled=True, round_index=0, previous=()):
    k = min(int(k), len(soft))
    uncertainty = posterior / (posterior + soft ** 2 + 1e-12)
    uncertainty = uncertainty / max(float(uncertainty.max()), 1e-12)
    correlation = np.abs(G) / np.sqrt(np.maximum(np.outer(np.diag(G), np.diag(G)), 1e-24))
    np.fill_diagonal(correlation, 0)
    chosen = []
    # On round two seed with the most uncertain previously EXCLUDED bit.
    available = [i for i in range(len(soft)) if i not in previous] if round_index else []
    if available:
        chosen.append(max(available, key=lambda i: uncertainty[i]))
    while len(chosen) < k:
        score = uncertainty.copy()
        if coupled and chosen:
            score += 0.5 * np.max(correlation[:, chosen], axis=1)
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return np.asarray(chosen)


def soft_probabilities(soft, posterior, *, temperature=1., epsilon=.1):
    if temperature <= 0 or not 0 < epsilon < .5:
        raise ValueError("Invalid confidence calibration")
    # Posterior Gaussian approximation; temperature fitted only on training labels.
    logits = 2 * soft / np.maximum(posterior * temperature, 1e-9)
    return np.clip(1 / (1 + np.exp(np.clip(logits, -40, 40))), epsilon, 1 - epsilon)


def exact_ml_reference(H, y):
    """Offline ONLY. Runtime hybrids never call or receive this oracle."""
    if H.shape[1] > 12:
        raise ValueError("Offline exact ML is limited to 12 users")
    G, b = quadratic(H, y)
    spins = spin_table(H.shape[1])
    objective = np.sum((spins @ G) * spins, axis=1) - 2 * (spins @ b)
    return spins[int(np.argmin(objective))].copy()
