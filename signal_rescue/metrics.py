"""Paired channel-group bootstrap and descriptive benchmark summaries."""

from __future__ import annotations
import numpy as np


def paired_bootstrap(errors, bits, groups, q_index, baseline_index, *, seed=7, resamples=2000):
    """Resample whole groups, retaining paired methods, SNRs, and noise repeats."""
    valid = np.isfinite(errors[:, q_index]) & np.isfinite(errors[:, baseline_index])
    errors, bits, groups = errors[valid], bits[valid], groups[valid]
    if not len(bits):
        raise ValueError("No paired observations")
    unique, inverse = np.unique(groups, return_inverse=True)
    q = np.bincount(inverse, weights=errors[:, q_index], minlength=len(unique))
    b = np.bincount(inverse, weights=errors[:, baseline_index], minlength=len(unique))
    n = np.bincount(inverse, weights=bits, minlength=len(unique))
    # Fixed-condition mixture: resample within seed/family/loading for controlled
    # channels, and whole physical tiles across ALL noise seeds for DeepMIMO.
    def stratum(group):
        parts = str(group).split(":")
        return ":".join(parts[:3]) if len(parts) >= 4 else (parts[0] if len(parts) > 1 else "all")
    strata = np.asarray([stratum(group) for group in unique])
    rng = np.random.default_rng(seed)
    sampled_q, sampled_b, sampled_bits = (np.zeros(resamples) for _ in range(3))
    for label in np.unique(strata):
        ids = np.flatnonzero(strata == label)
        weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=resamples)
        sampled_q += weights @ q[ids]
        sampled_b += weights @ b[ids]
        sampled_bits += weights @ n[ids]
    delta = (sampled_b - sampled_q) / sampled_bits
    gains = np.divide(sampled_b - sampled_q, sampled_b,
                      out=np.full(resamples, np.nan), where=sampled_b > 0)
    point_gain = float((b.sum() - q.sum()) / b.sum()) if b.sum() else None
    finite = gains[np.isfinite(gains)]
    return {"baseline_ber": float(b.sum() / n.sum()), "qaoa_ber": float(q.sum() / n.sum()),
            "ber_improvement": float((b.sum() - q.sum()) / n.sum()),
            "ber_improvement_ci95": np.quantile(delta, [.025, .975]).tolist(),
            "relative_ber_improvement": point_gain,
            "relative_improvement_ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "groups": len(unique), "channel_uses": len(bits), "bits": int(n.sum()),
            "bootstrap_resamples": resamples, "cluster": "whole channel / spatial-user group",
            "stratification": "controlled: seed/family/loading; DeepMIMO: physical tiles shared across noise seeds",
            "sign": "positive means QAOA has fewer bit errors", "ci_multiplicity": "unadjusted descriptive 95%"}


def summarize(data, methods, selection=None):
    rows = np.ones(len(data["bits"]), dtype=bool) if selection is None else np.asarray(selection)
    output = {}
    for i, name in enumerate(methods):
        keep = rows & np.isfinite(data["errors"][:, i])
        if not np.any(keep):
            continue
        errors, bits = data["errors"][keep, i], data["bits"][keep]
        timing = data["seconds"][keep, i]
        output[name] = {"channel_uses": int(keep.sum()), "bits": int(bits.sum()),
                        "bit_errors": int(errors.sum()), "ber": float(errors.sum() / bits.sum()),
                        "bler": float(np.mean(errors > 0)),
                        "mean_residual": float(np.mean(data["residual"][keep, i])),
                        "latency_s": {"mean": float(np.mean(timing)), "p95": float(np.quantile(timing, .95))}}
    return output
