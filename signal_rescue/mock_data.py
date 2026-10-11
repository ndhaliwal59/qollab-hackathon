"""Mock real-valued Rayleigh MIMO channels.

``load_instance`` is the only data boundary. Swap this for a DeepMIMO slice
later; detectors only need H, x, y, and the noise variance.
"""

from __future__ import annotations

import numpy as np

SCENARIOS = {
    "square": {"n_rx": 8, "n_tx": 8},
    "overloaded": {"n_rx": 8, "n_tx": 12},
}


def noise_variance(n_rx: int, n_tx: int, snr_db: float) -> float:
    """sigma^2 such that E[||Hx||^2] / E[||n||^2] equals the linear SNR.

    H is scaled by 1/sqrt(n_rx), so E[||Hx||^2] = n_tx for BPSK symbols.
    """
    snr = 10.0 ** (snr_db / 10.0)
    return n_tx / (n_rx * snr)


def load_instance(n_rx: int, n_tx: int, snr_db: float, rng: np.random.Generator):
    """One channel use: y = Hx + n, with x in {-1, +1}^n_tx."""
    H = rng.normal(0.0, 1.0 / np.sqrt(n_rx), size=(n_rx, n_tx))
    x = rng.choice(np.array([-1.0, 1.0]), size=n_tx)
    sigma2 = noise_variance(n_rx, n_tx, snr_db)
    noise = rng.normal(0.0, np.sqrt(sigma2), size=n_rx)
    y = H @ x + noise
    return {"H": H, "x": x, "y": y, "sigma2": sigma2, "snr_db": float(snr_db)}
