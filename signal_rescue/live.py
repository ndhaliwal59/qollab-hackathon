"""Reproducible single-frame experiment shared by localhost and Qollab."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import traceback

import numpy as np
from qiskit import QuantumCircuit

from signal_rescue.circuits import build_qaoa_circuit
from signal_rescue.channels import ChannelBank, LOADINGS, SNRS
from signal_rescue.core import cost, exact_ml_reference, mmse, ordered_sic
from signal_rescue.hybrid import detect, prepare
from signal_rescue.samplers import Config, SampleResult, sample


# Keep these paths as literals: Qollab discovers project data by scanning for
# plain path strings in Python source before it starts Pyodide.
CHANNEL_PATH = "data/deepmimo/asu_slice.npz"
PROVENANCE_PATH = "data/deepmimo/asu_slice.json"
FROZEN_PATH = "results/confirmation-full/frozen.json"
REPORT_PATH = "results/confirmation-full/report.json"
SOURCE_PATHS = ("signal_rescue/live.py", "signal_rescue/circuits.py", "signal_rescue/core.py",
                "signal_rescue/hybrid.py", "signal_rescue/samplers.py")

SUPPORTED_SHOTS = (64, 128)
SUPPORTED_SEEDS = (7, 8, 9, 10, 11)
FROZEN_PATH_VALUES: dict[str, object] | None = None
CHANNEL_BANKS: dict[int, ChannelBank] = {}


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"))


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def _frozen() -> dict:
    global FROZEN_PATH_VALUES
    if FROZEN_PATH_VALUES is None:
        frozen = _read_json(FROZEN_PATH)
        recorded_hash = frozen.get("freeze_sha256")
        payload = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
        actual_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if not recorded_hash or actual_hash != recorded_hash:
            raise ValueError("Frozen QAOA configuration failed its SHA-256 check")
        params = np.asarray(frozen.get("params"), dtype=float)
        config = Config(**frozen.get("config", {}))
        if config.p != 1 or config.k != 8 or config.warm or params.shape != (2,):
            raise ValueError("Live demo expects the frozen standard depth-1, 8-qubit model")
        if not np.all(np.isfinite(params)):
            raise ValueError("Frozen QAOA angles must be finite")
        FROZEN_PATH_VALUES = frozen
    return FROZEN_PATH_VALUES


def _bank(seed: int) -> ChannelBank:
    bank = CHANNEL_BANKS.get(seed)
    if bank is None:
        bank = ChannelBank(seed=seed, dataset="deepmimo", path=Path(CHANNEL_PATH))
        if len(CHANNEL_BANKS) >= len(SUPPORTED_SEEDS):
            CHANNEL_BANKS.clear()
        CHANNEL_BANKS[seed] = bank
    return bank


def _normalize_spec(spec: object) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("Experiment settings must be an object")
    users = spec.get("users", 12)
    seed = spec.get("seed", 7)
    scenario = spec.get("scenario_index", 0)
    snr = spec.get("snr_db", 12)
    shots = spec.get("shots_per_round", 64)
    request_id = spec.get("request_id", "")
    if users not in (8, 12):
        raise ValueError("Choose 8 or 12 concurrent users")
    if seed not in SUPPORTED_SEEDS:
        raise ValueError("Choose one of the reproducibility seeds 7–11")
    if type(scenario) is not int or not 0 <= scenario < 8:
        raise ValueError("Choose one of the eight held-out validation scenarios")
    if type(snr) is not int or snr not in SNRS:
        raise ValueError("SNR must be 0, 4, 8, 12, 16, or 20 dB")
    if shots not in SUPPORTED_SHOTS:
        raise ValueError("Shot budget must be 64 or 128 per correction round")
    if not isinstance(request_id, str) or len(request_id) > 96:
        raise ValueError("Request identifier must be a string of at most 96 characters")
    return {
        "request_id": request_id,
        "scenario_index": scenario,
        "seed": seed,
        "users": users,
        "loading": "square" if users == 8 else "overloaded",
        "snr_db": snr,
        "shots_per_round": shots,
    }


def _make_observation(spec: dict):
    return _bank(spec["seed"]).instance(
        "validation", "deepmimo", spec["loading"], spec["snr_db"], spec["scenario_index"]
    )


def _bits(symbols: np.ndarray) -> str:
    return "".join("1" if value < 0 else "0" for value in np.asarray(symbols).reshape(-1))


def _row(name: str, estimate: np.ndarray, truth: np.ndarray, H: np.ndarray,
         y: np.ndarray) -> dict:
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    errors = int(np.count_nonzero(estimate != truth))
    nbits = int(truth.size)
    return {
        "id": name,
        "bits": _bits(estimate),
        "errors": errors,
        "bits_total": nbits,
        "frame_ber": errors / nbits,
        "bler": int(errors > 0),
        "error_indices": np.flatnonzero(estimate != truth).astype(int).tolist(),
        "residual": cost(H, y, estimate),
    }


def _base_result(spec: dict) -> tuple[dict, object, Config, np.ndarray, tuple]:
    started = time.perf_counter()
    frozen = _frozen()
    config = Config(**frozen["config"])
    if spec["shots_per_round"] != config.shots:
        from dataclasses import replace
        config = replace(config, shots=spec["shots_per_round"])
    instance = _make_observation(spec)
    prepared = prepare(instance.H, instance.y, instance.sigma2, config)
    initial = prepared[-1]
    mmse_bits = mmse(instance.H, instance.y, instance.sigma2)[0]
    sic_bits = ordered_sic(instance.H, instance.y, instance.sigma2)
    uniform_histograms: list[list[dict]] = []

    def uniform_sampler(kind, h, J, warm_prob, sampler_config, params, sample_seed):
        result = sample("uniform", h, J, warm_prob, sampler_config, None, sample_seed)
        uniform_histograms.append(_histogram(result.candidates, h, J))
        return result

    uniform = detect(
        instance.H, instance.y, instance.sigma2, kind="uniform", config=config,
        params=None, seed=spec["seed"], prepared=prepared, replay=True,
        sampler_fn=uniform_sampler,
    )
    classical = _row("classical", initial, instance.truth, instance.H, instance.y)
    rows = [
        _row("mmse", mmse_bits, instance.truth, instance.H, instance.y),
        _row("mmse_sic", sic_bits, instance.truth, instance.H, instance.y),
        classical,
        _row("uniform", uniform.x, instance.truth, instance.H, instance.y),
    ]
    recipe = {
        "dataset": "DeepMIMO ASU campus 3.5 GHz, ray-traced, not over-the-air measured",
        "source_url": _bank(spec["seed"]).provenance["source_url"],
        "group": instance.group,
        "split": "validation (not confirmation test)",
        "seed": spec["seed"],
        "scenario_index": spec["scenario_index"],
        "users": spec["users"],
        "receive_antennas": 8,
        "snr_db": spec["snr_db"],
        "sigma2_complex": instance.sigma2,
        "csi": "perfect",
        "truth_bits": _bits(instance.truth),
        "truth_symbols": instance.truth.tolist(),
        "user_uncertainty": (prepared[4] / (prepared[4] + prepared[3] ** 2 + 1e-12)).tolist(),
        "channel_column_powers": (np.abs(instance.H) ** 2).sum(axis=0).tolist(),
        # Rows are receive antennas, columns are users: every antenna receives
        # every user's symbol. The scene uses these exact channel coefficients.
        "channel_link_powers": (np.abs(instance.H) ** 2).tolist(),
    }
    source_hash = _file_sha256(CHANNEL_PATH)
    model = {
        "frozen_sha256": frozen["freeze_sha256"],
        "dataset_sha256": source_hash,
        "source_sha256": {path: _file_sha256(path) for path in SOURCE_PATHS},
        "config": config.dictionary(),
        "angles": [float(v) for v in frozen["params"]],
        "circuit_engine": "Qiskit QuantumCircuit on the supplied Qiskit backend",
        "local_backend": "Qiskit backend supplied by runtime",
    }
    result = {
        "request_id": spec["request_id"],
        "recipe": recipe,
        "model": model,
        "truth_bits": recipe["truth_bits"],
        "methods": rows,
        "uniform_rounds": uniform_histograms,
        "uniform_telemetry": uniform.telemetry,
        "uniform_replay": uniform.replay,
        "preview_seconds": time.perf_counter() - started,
    }
    return result, instance, config, np.asarray(frozen["params"], dtype=float), prepared


def _histogram(candidates: np.ndarray, h: np.ndarray, J: np.ndarray) -> list[dict]:
    unique, frequencies = np.unique(np.asarray(candidates, dtype=float), axis=0, return_counts=True)
    energies = unique @ h + np.sum((unique @ np.triu(J, 1)) * unique, axis=1)
    order = sorted(range(len(unique)), key=lambda i: (-int(frequencies[i]), float(energies[i])))
    return [{
        "bits": "".join("1" if value < 0 else "0" for value in unique[i, ::-1]),
        "count": int(frequencies[i]),
        "energy": float(energies[i]),
    } for i in order]


def _decode_counts(counts: dict, k: int, shots: int, h: np.ndarray, J: np.ndarray):
    """Expand Qiskit counts in little-endian qubit order used by the Ising block."""
    expanded: list[np.ndarray] = []
    histogram = []
    for raw_bits, count in counts.items():
        packed = int(str(raw_bits).replace(" ", ""), 2)
        spins = 1.0 - 2.0 * ((packed >> np.arange(k)) & 1)
        count = int(count)
        expanded.extend([spins.copy() for _ in range(count)])
        histogram.append({
            "bits": "".join("1" if value < 0 else "0" for value in spins[::-1]),
            "count": count,
            "energy": float(spins @ h + np.sum((spins @ np.triu(J, 1)) * spins)),
        })
    if len(expanded) != shots:
        raise RuntimeError(f"Backend returned {len(expanded)} samples for {shots} requested shots")
    return np.asarray(expanded, dtype=float), sorted(histogram, key=lambda row: (-row["count"], row["energy"]))


def _qiskit_sample(h: np.ndarray, J: np.ndarray, config: Config, params: np.ndarray,
                   seed: int, backend, local: bool) -> SampleResult:
    started = time.perf_counter()
    scale = max(float(np.max(np.abs(h))), float(np.max(np.abs(J))), 1e-12)
    circuit, gamma, beta = build_qaoa_circuit(h / scale, J / scale, config.p)
    bindings = {gamma[layer]: float(params[layer]) for layer in range(config.p)}
    bindings.update({beta[layer]: float(params[config.p + layer]) for layer in range(config.p)})
    measured = circuit.assign_parameters(bindings)
    measured.measure_all()
    try:
        if local:
            job = backend.run(measured, shots=config.shots, seed_simulator=int(seed))
        else:
            job = backend.run(measured, shots=config.shots)
        counts = job.result().get_counts()
    except Exception as error:
        raise RuntimeError(f"Qiskit backend execution failed: {type(error).__name__}: {error}") from error

    candidates, candidate_histogram = _decode_counts(counts, len(h), config.shots, h, J)
    telemetry = {
        "sampler": "qaoa", "shots": config.shots, "circuit_invocations": 1,
        "candidates": config.shots, "unique_candidates": len(counts),
        "sa_spin_proposals": 0, "seconds": time.perf_counter() - started,
        "circuit_depth": measured.depth(),
        "gate_summary": {str(gate): int(count) for gate, count in measured.count_ops().items()},
        # Keep the problem-dependent structure so the browser can draw the
        # circuit that actually produced these samples (rather than a generic
        # QAOA illustration).
        "field_qubits": [i for i, value in enumerate(h) if value != 0.0],
        "coupling_pairs": [[i, j] for i in range(len(h)) for j in range(i + 1, len(h))
                           if J[i, j] != 0.0],
        "layer_angles": [{"gamma": float(params[layer]), "beta": float(params[config.p + layer])}
                         for layer in range(config.p)],
    }
    telemetry["measurement_counts"] = candidate_histogram
    return SampleResult(candidates, telemetry)


def _compute(spec: dict, backend=None, local=False) -> dict:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    result, instance, config, params, prepared = _base_result(spec)
    result["model"]["local_backend"] = "BasicSimulator" if local else "Qollab-selected backend"
    qaoa_histograms: list[list[dict]] = []

    def qaoa_sampler(kind, h, J, warm_prob, sampler_config, frozen_params, sample_seed):
        if backend is None:
            raise RuntimeError("A Qiskit execution backend was not supplied")
        sample_result = _qiskit_sample(h, J, sampler_config, params, sample_seed, backend, local)
        qaoa_histograms.append(sample_result.telemetry["measurement_counts"])
        return SampleResult(sample_result.candidates, {
            key: value for key, value in sample_result.telemetry.items() if key != "measurement_counts"
        })

    qaoa = detect(
        instance.H, instance.y, instance.sigma2, kind="qaoa", config=config,
        params=params, seed=spec["seed"], prepared=prepared, replay=True,
        sampler_fn=qaoa_sampler,
    )
    result["methods"].append(_row("qaoa", qaoa.x, instance.truth, instance.H, instance.y))
    # Exact ML is evaluated after both detector paths finish. It never enters either.
    ml_started = time.perf_counter()
    ml = exact_ml_reference(instance.H, instance.y)
    result["methods"].append(_row("ml_reference", ml, instance.truth, instance.H, instance.y))
    result["ml_reference_seconds"] = time.perf_counter() - ml_started
    result["qaoa_rounds"] = qaoa_histograms
    result["qaoa_replay"] = qaoa.replay
    result["qaoa_telemetry"] = qaoa.telemetry
    result["comparison"] = {
        "same_observation": True,
        "shots_per_method_per_round": config.shots,
        "rounds": config.rounds,
        "qaoa_total_shots": qaoa.telemetry["shots"],
        "uniform_total_candidates": result["uniform_telemetry"]["candidates"],
        "budgets_matched": (qaoa.telemetry["shots"] ==
                             result["uniform_telemetry"]["candidates"]),
        "selection_note": "Every sampler uses the same measured objective and common classical refinement.",
    }
    result["execution"] = {
        "request_id": spec["request_id"],
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "reproducible_seed": spec["seed"],
        "sampling_seeded": bool(local),
        "status": "partial" if qaoa.telemetry["fallbacks"] else "complete",
        "backend": result["model"]["local_backend"],
        "circuit_invocations": qaoa.telemetry["circuit_invocations"],
        "shots": qaoa.telemetry["shots"],
        "fallbacks": qaoa.telemetry["fallbacks"],
        "warning": ("One or more circuit executions failed; QAOA output uses the preserved classical incumbent."
                    if qaoa.telemetry["fallbacks"] else None),
    }
    result["experiment_seconds"] = time.perf_counter() - started
    return result


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dispatch(backend, action: str, payload_json: str) -> str:
    """JSON-only boundary callable from a local server or Qollab's JS bridge."""
    try:
        if action == "bootstrap":
            frozen = _frozen()
            report = _read_json(REPORT_PATH)
            provenance = _read_json(PROVENANCE_PATH)
            return _json_dumps({
                "status": "ready",
                "supported": {"users": [8, 12], "receive_antennas": 8,
                              "snr_db": list(SNRS), "seeds": list(SUPPORTED_SEEDS),
                              "scenarios": 8, "shots_per_round": list(SUPPORTED_SHOTS)},
                "dataset": {"name": "DeepMIMO ASU campus, 3.5 GHz", "measured": False,
                            "provenance": {**provenance, "sha256": _file_sha256(CHANNEL_PATH)},
                            "sha256": _file_sha256(CHANNEL_PATH)},
                "model": {"freeze_sha256": frozen["freeze_sha256"],
                          "config": frozen["config"], "params": frozen["params"]},
                "benchmark": {"summary": report["primary_summary"],
                              "comparisons": report["paired_primary_comparisons"]["all"],
                              "success_gates": report["success_gates"],
                              "conclusion": report["conclusion"],
                              "primary_snrs_db": report["primary_snrs_db"],
                              "primary_channel_uses": 32000,
                              "all_channel_uses": 144000},
            })
        payload = json.loads(payload_json or "{}")
        spec = _normalize_spec(payload)
        if action == "preview":
            if not spec["request_id"]:
                raise ValueError("A preview request identifier is required")
            result, *_ = _base_result(spec)
            result.pop("uniform_replay", None)
            return _json_dumps({"status": "ok", "result": result})
        if action == "run":
            if backend is None:
                raise ValueError("Run requires a Qiskit backend")
            if not spec["request_id"]:
                raise ValueError("A quantum-run request identifier is required")
            result = _compute(spec, backend=backend, local=bool(getattr(backend, "name", "") == "basic_simulator"))
            result.pop("uniform_replay", None)
            return _json_dumps({"status": "ok", "result": result})
        raise ValueError("Unknown action. Use bootstrap, preview, or run.")
    except Exception as error:
        message = str(error)
        if len(message) > 200:
            message = message[:197] + "..."
        return _json_dumps({"status": "error", "error": message,
                            "type": type(error).__name__})
