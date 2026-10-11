"""Small-statevector QAOA and budget-matched classical candidate samplers.

The simulator evolves all amplitudes, as any exact simulator must, but returns
ONLY finite-shot samples. It never searches the spectrum for the optimum.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import time
import numpy as np
from signal_rescue.core import energies, spin_table


@dataclass(frozen=True)
class Config:
    k: int = 6
    p: int = 2
    shots: int = 64
    warm: bool = True
    epsilon: float = .1
    objective: str = "cvar"
    alpha: float = .25
    rounds: int = 2
    finalists: int = 4
    refine_steps: int = 8
    sa_sweeps: int = 12
    coupled: bool = True
    temperature: float = 1.

    def __post_init__(self):
        if (self.k not in (4, 6, 8) or self.p not in (1, 2, 3) or self.shots not in (64, 128)
                or not 0 < self.epsilon < .5 or self.objective not in ("mean", "cvar")
                or not 0 < self.alpha <= 1 or self.rounds not in (1, 2)
                or not 1 <= self.finalists <= self.shots or not 0 <= self.refine_steps <= 32
                or not 1 <= self.sa_sweeps <= 64 or not np.isfinite(self.temperature) or self.temperature <= 0):
            raise ValueError("Invalid or unbounded hybrid configuration")

    def dictionary(self):
        return asdict(self)


@dataclass
class SampleResult:
    candidates: np.ndarray
    telemetry: dict


def spectrum_for_simulation(h, J):
    """Cost diagonal for phase evolution, NOT an oracle or candidate list."""
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(J))), 1e-12)
    return energies(spin_table(len(h)), h / scale, J / scale)


def probabilities(h, J, params, *, p, warm_prob=None):
    h, J, params = np.asarray(h), np.asarray(J), np.asarray(params, dtype=float)
    k = len(h)
    if (not 1 <= k <= 8 or J.shape != (k, k) or params.shape != (2 * p,)
            or not np.all(np.isfinite(params)) or not np.all(np.isfinite(h))
            or not np.all(np.isfinite(J))):
        raise ValueError("Invalid QAOA dimensions or parameters")
    q = np.full(k, .5) if warm_prob is None else np.asarray(warm_prob)
    if q.shape != (k,) or np.any(q <= 0) or np.any(q >= 1) or not np.all(np.isfinite(q)):
        raise ValueError("Warm-start probabilities must lie strictly between zero and one")
    bits = (1 - spin_table(k)) / 2
    state = np.sqrt(np.prod(np.where(bits, q, 1 - q), axis=1)).astype(complex)
    diagonal = spectrum_for_simulation(h, J)
    for layer in range(p):
        state *= np.exp(-1j * params[layer] * diagonal)
        beta = params[p + layer]
        c, s = np.cos(beta), np.sin(beta)
        for bit in range(k):
            view = state.reshape(-1, 2, 1 << bit)
            low, high = view[:, 0, :].copy(), view[:, 1, :].copy()
            if warm_prob is None:
                view[:, 0, :] = c * low - 1j * s * high
                view[:, 1, :] = c * high - 1j * s * low
            else:
                z, x = 1 - 2 * q[bit], 2 * np.sqrt(q[bit] * (1 - q[bit]))
                view[:, 0, :] = (c + 1j * s * z) * low + 1j * s * x * high
                view[:, 1, :] = 1j * s * x * low + (c - 1j * s * z) * high
    probs = np.abs(state) ** 2
    return probs / probs.sum()


def objective_value(energy, probabilities_, *, objective="mean", alpha=.25):
    if objective == "mean":
        return float(energy @ probabilities_)
    # Fractional mass at the CVaR boundary, rather than biasing by ceil(shots*alpha).
    order = np.argsort(energy)
    weight = probabilities_[order]
    take = np.minimum(weight, np.maximum(alpha - np.cumsum(weight) + weight, 0))
    return float(energy[order] @ take / alpha)


def sample(kind, h, J, warm_prob, config: Config, params, seed) -> SampleResult:
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    k = len(h)
    telemetry = {"sampler": kind, "candidates": config.shots, "shots": 0,
                 "circuit_invocations": 0, "optimizer_evaluations": 0, "sa_spin_proposals": 0}
    if kind == "qaoa":
        probs = probabilities(h, J, params, p=config.p,
                              warm_prob=warm_prob if config.warm else None)
        masks = rng.choice(1 << k, size=config.shots, p=probs)
        candidates = 1. - 2. * ((masks[:, None] >> np.arange(k)) & 1)
        telemetry.update(shots=config.shots, circuit_invocations=1)
    elif kind in ("uniform", "biased", "sa"):
        q = np.full(k, .5) if kind == "uniform" else warm_prob
        candidates = 1. - 2. * (rng.random((config.shots, k)) < q)
        if kind == "sa":
            scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(J))), 1e-12)
            for temperature in np.geomspace(2 * scale, .05 * scale, config.sa_sweeps):
                for bit in rng.permutation(k):
                    change = -2 * candidates[:, bit] * (h[bit] + candidates @ J[bit])
                    accept = np.log(np.maximum(rng.random(config.shots), 1e-300)) < -change / temperature
                    candidates[accept, bit] *= -1
            telemetry["sa_spin_proposals"] = config.shots * k * config.sa_sweeps
    else:
        raise ValueError(f"Unknown candidate sampler: {kind}")
    telemetry["unique_candidates"] = len(np.unique(candidates, axis=0))
    telemetry["seconds"] = time.perf_counter() - started
    return SampleResult(candidates.astype(float), telemetry)
