"""DEPRECATED exploratory mMobile axis-remapping adapter. NOT a MIMO dataset.

The public mMobile files contain complex 28 GHz CSI measured over 256
subcarriers, 31 beams, and several UE positions. The detector benchmark is a
real-valued BPSK model, so each measured CSI slice is converted to a normalized
real channel matrix and evaluated across a controlled synthetic SNR sweep.
Beams and subcarriers are not simultaneous antenna/user dimensions. These
outputs cannot support a scored MIMO detection or quantum-contribution claim.
Use the dimensioned DeepMIMO slice and controlled complex models in channels.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MMOBILE_DIR = ROOT / "data" / "mmobile"
SCENARIOS = {
    "square": {"n_rx": 8, "n_tx": 8},
    "overloaded": {"n_rx": 8, "n_tx": 12},
}
SCENARIO_FILES = {
    "square": (
        "indoor_2_5m_data.mat",
        "indoor_4_2m_data.mat",
        "outdoor_10m_data.mat",
        "outdoor_30m_data.mat",
    ),
    "overloaded": (
        "indoor_2_5m_data.mat",
        "indoor_4_2m_data.mat",
        "outdoor_10m_data.mat",
        "outdoor_30m_data.mat",
    ),
}
SOURCE_METADATA = {
    "scored_mimo_eligible": False,
    "status": "deprecated exploratory axis remapping; excluded from v2 benchmark",
    "name": "ucsd-mmobile-csi",
    "dataset": "mMobile Mobility Dataset",
    "carrier": "28 GHz",
    "subcarriers": 256,
    "beams": 31,
    "files": 4,
    "channel_data": "measured complex CSI",
    "evaluation_note": "NOT physical MIMO: beam/subcarrier axes were remapped to antennas; exploratory only",
}


def noise_variance(n_rx: int, n_tx: int, snr_db: float) -> float:
    snr = 10.0 ** (snr_db / 10.0)
    return n_tx / (n_rx * snr)


@lru_cache(maxsize=None)
def _load_csi(filename: str) -> np.ndarray:
    path = MMOBILE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"mMobile CSI file is missing: {path}")
    data = loadmat(path)
    csi = np.asarray(data["rx_h_est_vec"], dtype=np.complex128)
    if csi.ndim != 3 or csi.shape[1] < 12 or csi.shape[2] < 8:
        raise ValueError(f"Unexpected CSI shape in {path}: {csi.shape}")
    return csi


def load_instance(
    n_rx: int,
    n_tx: int,
    snr_db: float,
    rng: np.random.Generator,
):
    """Build one detection instance from a measured CSI slice."""
    scenario = "overloaded" if n_tx > n_rx else "square"
    filename = str(rng.choice(SCENARIO_FILES[scenario]))
    csi = _load_csi(filename)
    position = int(rng.integers(0, csi.shape[0]))
    beam_start = int(rng.integers(0, csi.shape[1] - n_tx + 1))
    sub_start = int(rng.integers(0, csi.shape[2] - n_rx + 1))
    # Stored as [position, beam, subcarrier]; transpose to [receive, transmit].
    measured = csi[position, beam_start : beam_start + n_tx, sub_start : sub_start + n_rx].T
    H = np.real(measured)
    norm = float(np.linalg.norm(H))
    if norm <= 1e-12:
        H = np.ones((n_rx, n_tx), dtype=float)
        norm = float(np.linalg.norm(H))
    H = H * np.sqrt(n_tx) / norm
    x = rng.choice(np.array([-1.0, 1.0]), size=n_tx)
    sigma2 = noise_variance(n_rx, n_tx, snr_db)
    y = H @ x + rng.normal(0.0, np.sqrt(sigma2), size=n_rx)
    return {"H": H, "x": x, "y": y, "sigma2": sigma2, "snr_db": float(snr_db)}
