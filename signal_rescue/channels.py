"""Dimensioned, complex BPSK channels with group-disjoint experiment splits.

Synthetic channels are *controlled models*, not measured 5G traffic. The old
mMobile beam/subcarrier-to-antenna mapping is deliberately not used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

FAMILIES = ("rayleigh", "correlated", "rician")
LOADINGS = {"square": (8, 8), "overloaded": (8, 12)}
SNRS = (0, 4, 8, 12, 16, 20)
PRIMARY_SNRS = (8, 12, 16, 20)
CSI_LEVELS = ("perfect", "-20", "-10")
DOWNLOAD_CAP = 100_000_000
PROCESSED_CAP = 20_000_000


def keyed_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def split_groups(groups, seed: int) -> dict[str, list[str]]:
    """Split whole channel/spatial groups BEFORE making symbols or noise."""
    unique = sorted(set(map(str, groups)))
    if len(unique) < 10:
        raise ValueError("At least ten independent channel/spatial groups are required")
    order = np.random.default_rng(keyed_seed("split", seed)).permutation(len(unique))
    a, b = int(0.6 * len(order)), int(0.8 * len(order))
    return {name: [unique[i] for i in part] for name, part in
            zip(("train", "validation", "test"), (order[:a], order[a:b], order[b:]))}


def complex_normal(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)


def model_channel(family: str, n_rx: int, n_tx: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffuse = complex_normal(rng, (n_rx, n_tx)) / np.sqrt(n_rx)
    if family == "rayleigh":
        channel = diffuse
    elif family == "correlated":
        rx = 0.5 ** np.abs(np.arange(n_rx)[:, None] - np.arange(n_rx))
        tx = 0.65 ** np.abs(np.arange(n_tx)[:, None] - np.arange(n_tx))
        channel = np.linalg.cholesky(rx) @ diffuse @ np.linalg.cholesky(tx).T
    elif family == "rician":
        angles = rng.uniform(-1, 1, n_tx)
        los = np.exp(1j * np.pi * np.arange(n_rx)[:, None] * angles) / np.sqrt(n_rx)
        los *= np.exp(1j * rng.uniform(-np.pi, np.pi, n_tx))
        channel = np.sqrt(4 / 5) * los + np.sqrt(1 / 5) * diffuse
    else:
        raise ValueError(f"Unknown channel family: {family}")
    # One global normalization: do NOT normalize columns and erase near/far power.
    channel *= 10 ** (rng.uniform(-6, 6, n_tx) / 20)
    return channel * np.sqrt(n_tx / np.vdot(channel, channel).real)


@dataclass
class Instance:
    H: np.ndarray  # receiver CSI, possibly imperfect
    y: np.ndarray
    truth: np.ndarray
    sigma2: float  # E[|complex noise|^2], not variance of each real component
    group: str
    condition: float


class ChannelBank:
    def __init__(self, *, seed=7, dataset="synthetic", path=None, groups=300):
        self.seed, self.dataset = int(seed), dataset
        self.cache: dict[tuple, np.ndarray] = {}
        self.real_channels = None
        if dataset == "synthetic":
            self.groups = [f"channel-{i:04d}" for i in range(groups)]
            self.provenance = {"kind": "controlled_complex_channel_models", "families": list(FAMILIES),
                               "measured": False, "download_bytes": 0,
                               "normalization": "one Frobenius scalar; relative user powers retained"}
        elif dataset == "deepmimo":
            self._load_slice(Path(path) if path else None)
        else:
            raise ValueError("dataset must be synthetic or deepmimo")
        # Fixed physical locations cannot change split when the noise seed changes.
        self.splits = split_groups(self.groups, 7 if self.real_channels is not None else self.seed)

    def _load_slice(self, path):
        if path is None or not path.is_file():
            raise ValueError("DeepMIMO requires an explicit local, provenance-tagged NPZ slice")
        if path.stat().st_size > PROCESSED_CAP:
            raise ValueError("Processed dataset exceeds the 20 MB cap")
        manifest_path = path.with_suffix(".json")
        if not manifest_path.is_file():
            raise ValueError("DeepMIMO slice requires a matching provenance .json")
        manifest = json.loads(manifest_path.read_text())
        for key in ("source_url", "scenario", "subcarrier", "bs_antennas", "ue_antennas",
                    "spatial_grouping", "download_bytes"):
            if key not in manifest:
                raise ValueError(f"Missing dataset provenance: {key}")
        if manifest["bs_antennas"] != 8 or manifest["ue_antennas"] != 1:
            raise ValueError("Expected 8 BS antennas and single-antenna users")
        if not 0 <= manifest["download_bytes"] <= DOWNLOAD_CAP:
            raise ValueError("Source download exceeds the 100 MB cap")
        if not manifest["spatial_grouping"]:
            raise ValueError("Adjacent user positions must share a spatial group")
        # Check expanded size before loading a compressed NPZ (zip-bomb defense).
        from zipfile import ZipFile
        with ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > PROCESSED_CAP:
                raise ValueError("Expanded dataset exceeds the 20 MB cap")
        with np.load(path, allow_pickle=False) as data:
            channels, group_ids = data["H"], data["group_ids"]
            if "user_ids" in data:
                owners = {}
                for row, group in zip(data["user_ids"], group_ids):
                    for user in row:
                        if int(user) in owners and owners[int(user)] != str(group):
                            raise ValueError("A physical user position leaks across spatial groups")
                        owners[int(user)] = str(group)
        if (channels.ndim != 3 or channels.shape[1:] != (8, 12)
                or not np.iscomplexobj(channels) or not np.all(np.isfinite(channels))
                or len(group_ids) != len(channels) or group_ids.ndim != 1
                or np.any(np.linalg.norm(channels, axis=(1, 2)) == 0)):
            raise ValueError("Expected finite complex H[realizations, 8, 12] and group_ids")
        self.real_channels = channels
        self.group_rows = {str(g): np.flatnonzero(group_ids.astype(str) == str(g))
                           for g in np.unique(group_ids)}
        self.groups = list(self.group_rows)
        self.provenance = {**manifest, "kind": "DeepMIMO_ray_traced_channels", "measured": False,
                           "processed_bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def instance(self, split, family, loading, snr, index, *, csi="perfect") -> Instance:
        if split not in self.splits or loading not in LOADINGS or csi not in CSI_LEVELS:
            raise ValueError("Invalid split, loading, or CSI error level")
        if not np.isfinite(snr) or index < 0:
            raise ValueError("SNR must be finite and index nonnegative")
        ids = self.splits[split]
        group = ids[index % len(ids)]
        repeat = index // len(ids)
        n_rx, n_tx = LOADINGS[loading]
        key = (family, loading, group)
        if self.real_channels is not None:
            rows = self.group_rows[group]
            H = self.real_channels[rows[repeat % len(rows)], :, :n_tx].copy()
            H *= np.sqrt(n_tx / np.vdot(H, H).real)
        else:
            if key not in self.cache:
                self.cache[key] = model_channel(family, n_rx, n_tx,
                                               keyed_seed("H", self.seed, *key))
            H = self.cache[key]
        # Same truth/noise across CSI conditions; completely separate RNG from H.
        rng = np.random.default_rng(keyed_seed("symbols-noise", self.seed, split,
                                              family, loading, group, snr, repeat))
        truth = rng.choice([-1., 1.], n_tx)
        signal_power = np.vdot(H, H).real / n_rx
        sigma2 = float(signal_power / 10 ** (float(snr) / 10))
        y = H @ truth + np.sqrt(sigma2) * complex_normal(rng, n_rx)
        estimate = H.copy()
        if csi != "perfect":
            error_rng = np.random.default_rng(keyed_seed("CSI", self.seed, family, loading, group, csi))
            error = complex_normal(error_rng, H.shape)
            error *= np.sqrt(np.vdot(H, H).real * 10 ** (float(csi) / 10) / np.vdot(error, error).real)
            estimate += error
        real_H = np.concatenate((estimate.real, estimate.imag), axis=0)
        group_id = (f"deepmimo:{group}" if self.real_channels is not None else
                    f"{self.seed}:{family}:{loading}:{group}")
        return Instance(estimate, y, truth, sigma2, group_id,
                        float(np.linalg.cond(real_H)))

    def manifest(self):
        return {"seed": self.seed, "provenance": self.provenance, "splits": self.splits,
                "split_unit": "whole channel / spatial-user group; all SNRs and repeats stay together",
                "split_ratio": [0.6, 0.2, 0.2], "awgn": "circular complex; each real component has sigma2/2"}


class ExperimentBank:
    """Route families to controlled models or the same fixed spatial dataset."""
    def __init__(self, seed, *, dataset="both", path=None):
        self.seed = seed
        self.controlled = ChannelBank(seed=seed) if dataset in ("controlled", "both") else None
        self.real = ChannelBank(seed=seed, dataset="deepmimo", path=path) if dataset in ("deepmimo", "both") else None

    def instance(self, split, family, *args, **kwargs):
        bank = self.real if family == "deepmimo" else self.controlled
        if bank is None:
            raise ValueError(f"Dataset for {family} is unavailable")
        return bank.instance(split, family, *args, **kwargs)

    def manifest(self):
        return {"seed": self.seed,
                "controlled": self.controlled.manifest() if self.controlled else None,
                "deepmimo": self.real.manifest() if self.real else None}
