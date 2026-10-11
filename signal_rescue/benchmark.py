"""Reproducible v2 5G hybrid benchmark. Run with --profile smoke before --profile full."""

from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from signal_rescue.channels import (CSI_LEVELS, FAMILIES, LOADINGS, PRIMARY_SNRS,
                                    SNRS, ExperimentBank, keyed_seed)
from signal_rescue.core import cost, exact_ml_reference, mmse, ordered_sic, quadratic, refine
from signal_rescue.hybrid import detect, prepare
from signal_rescue.metrics import paired_bootstrap, summarize
from signal_rescue.samplers import Config
from signal_rescue.training import train_and_freeze

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("mmse", "mmse_sic", "local_single", "classical", "uniform", "biased", "sa", "qaoa",
           "ml_reference", "opposite_warm", "uncertain_only", "no_refinement", "real_only")
MATCHED = ("uniform", "biased", "sa")
COMPARATORS = ("classical", *MATCHED)
PROFILES = {"smoke": {"seeds": [7, 8], "uses_per_seed": 4, "configs": 4, "max_evals": 10,
                       "starts": 1, "training_uses": 24, "validation_uses": 24, "workers": 1},
            "full": {"seeds": [7, 8, 9, 10, 11], "uses_per_seed": 200, "configs": 24,
                     "max_evals": 50, "starts": 3, "training_uses": 96, "validation_uses": 480, "workers": 4}}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def verify_freeze(frozen):
    payload = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if frozen.get("freeze_sha256") != expected:
        raise ValueError("Frozen training artifact failed its SHA-256 integrity check")


def timed(function, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def evaluate_cell(banks, family, loading, snr, csi, uses_per_seed, frozen, *, ablations=True):
    config, params = Config(**frozen["config"]), np.asarray(frozen["params"])
    opposite, opposite_params = Config(**frozen["opposite_warm_config"]), np.asarray(frozen["opposite_warm_params"])
    n_rows, n_methods = len(banks) * uses_per_seed, len(METHODS)
    arrays = {"errors": np.full((n_rows, n_methods), np.nan, np.float32),
              "seconds": np.full((n_rows, n_methods), np.nan, np.float32),
              "residual": np.full((n_rows, n_methods), np.nan, np.float32),
              "bits": np.full(n_rows, LOADINGS[loading][1], np.int16),
              "groups": np.empty(n_rows, dtype="U100"),
              "condition": np.empty(n_rows, np.float32),
              "seed": np.empty(n_rows, np.int32)}
    counters = {name: {key: 0 for key in ("shots", "circuit_invocations", "candidates", "candidate_refinements",
                                         "sa_spin_proposals", "fallbacks", "residual_violations")}
                for name in METHODS}
    replay = None
    for bank_index, bank in enumerate(banks):
        for index in range(uses_per_seed):
            row = bank_index * uses_per_seed + index
            instance = bank.instance("test", family, loading, snr, index, csi=csi)
            H, y, sigma2 = instance.H, instance.y, instance.sigma2
            arrays["groups"][row], arrays["condition"][row], arrays["seed"][row] = instance.group, instance.condition, bank.seed

            def record(name, x, seconds):
                col = METHODS.index(name)
                arrays["errors"][row, col] = np.count_nonzero(x != instance.truth)
                arrays["residual"][row, col] = cost(H, y, x)
                arrays["seconds"][row, col] = seconds

            (hard, _, _), mmse_seconds = timed(mmse, H, y, sigma2)
            record("mmse", hard, mmse_seconds)
            sic, seconds = timed(ordered_sic, H, y, sigma2)
            record("mmse_sic", sic, seconds)
            prepared, preparation_seconds = timed(prepare, H, y, sigma2, config)
            G, b = prepared[:2]
            record("classical", prepared[-1], preparation_seconds)
            single, seconds = timed(refine, G, b, hard, steps=config.refine_steps, pair=False)
            record("local_single", single[0][0], seconds + mmse_seconds)
            for kind in (*MATCHED, "qaoa"):
                result = detect(H, y, sigma2, kind=kind, config=config, params=params,
                                seed=keyed_seed("test-sampler", bank.seed, family, loading, snr, index, kind),
                                prepared=prepared, replay=kind == "qaoa" and row == 0 and snr == 12 and csi == "perfect")
                record(kind, result.x, result.telemetry["seconds"] + preparation_seconds)
                for key in counters[kind]:
                    counters[kind][key] += (int(not result.telemetry["residual_nonincreasing"])
                                            if key == "residual_violations" else result.telemetry[key])
                if result.replay is not None:
                    replay = {"family": family, "loading": loading, "snr_db": snr,
                              "truth": instance.truth.tolist(), **result.replay}
            # Exact ML is an offline reference and is deliberately computed AFTER hybrids.
            reference, seconds = timed(exact_ml_reference, H, y)
            record("ml_reference", reference, seconds)
            if ablations and csi == "perfect":
                for name, ablation_config, angles in (
                    ("opposite_warm", opposite, opposite_params),
                    ("uncertain_only", replace(config, coupled=False), params),
                    ("no_refinement", replace(config, refine_steps=0), params)):
                    result = detect(H, y, sigma2, config=ablation_config, params=angles,
                                    seed=keyed_seed("test-sampler", bank.seed, family, loading, snr, index, "qaoa"))
                    record(name, result.x, result.telemetry["seconds"])
                    for key in counters[name]:
                        counters[name][key] += (int(not result.telemetry["residual_nonincreasing"])
                                                if key == "residual_violations" else result.telemetry[key])
                result = detect(H.real, y.real, sigma2 / 2, config=config, params=params,
                                seed=keyed_seed("test-sampler", bank.seed, family, loading, snr, index, "qaoa"))
                record("real_only", result.x, result.telemetry["seconds"])
                for key in counters["real_only"]:
                    counters["real_only"][key] += (int(not result.telemetry["residual_nonincreasing"])
                                                  if key == "residual_violations" else result.telemetry[key])
    return arrays, counters, replay


def build_report(data, cell_reports, frozen, settings, counters, wall_seconds, first_invocation_s, cold_probe, warmup_cost):
    primary = np.isin(data["snr"], PRIMARY_SNRS) & (data["csi"] == "perfect")
    q = METHODS.index("qaoa")
    comparisons = {}
    for loading in ("all", *LOADINGS):
        keep = primary & ((data["loading"] == loading) if loading != "all" else True)
        comparisons[loading] = {method: paired_bootstrap(data["errors"][keep], data["bits"][keep],
                                                         data["groups"][keep], q, METHODS.index(method))
                                for method in COMPARATORS}
    robustness = {}
    for loading in LOADINGS:
        robustness[loading] = {}
        for csi in CSI_LEVELS:
            keep = np.isin(data["snr"], PRIMARY_SNRS) & (data["csi"] == csi) & (data["loading"] == loading)
            baseline = frozen["classical_baseline_by_loading"][loading]
            robustness[loading][csi] = paired_bootstrap(data["errors"][keep], data["bits"][keep],
                                                        data["groups"][keep], q, METHODS.index(baseline))
    primary_summary = summarize(data, METHODS, primary)
    strongest = min(COMPARATORS, key=lambda name: primary_summary[name]["ber"])
    adequate = len(settings["seeds"]) >= 5 and min(c["channel_uses"] for c in cell_reports) >= 1000
    gain = comparisons["all"][strongest]["relative_ber_improvement"]
    gate = {"at_least_5_seeds_and_1000_uses_per_cell": adequate,
            "relative_ber_gain_at_least_5pct": gain is not None and gain >= .05,
            "paired_ci_excludes_zero_vs_every_classical_hybrid":
                all(value["ber_improvement_ci95"][0] > 0 for value in comparisons["all"].values()),
            "no_supported_aggregate_regression_per_loading_or_csi":
                all(value["ber_improvement_ci95"][1] >= 0 for levels in robustness.values() for value in levels.values()),
            "warm_simulator_p95_under_1s": primary_summary["qaoa"]["latency_s"]["p95"] < 1,
            "no_sampler_failures_or_residual_regressions": all(
                counters["qaoa"][key] == 0 for key in ("fallbacks", "residual_violations"))}
    conditioning = {name: summarize(data, METHODS, primary & keep) for name, keep in
                    (("condition_lt_3", data["condition"] < 3),
                     ("condition_3_to_10", (data["condition"] >= 3) & (data["condition"] < 10)),
                     ("condition_ge_10", data["condition"] >= 10))}
    return {"schema_version": 2, "settings": settings, "freeze_sha256": frozen["freeze_sha256"],
            "primary_snrs_db": PRIMARY_SNRS, "primary_csi": "perfect", "methods": METHODS,
            "primary_summary": primary_summary, "cells": cell_reports,
            "primary_by_family": {family: summarize(data, METHODS, primary & (data["family"] == family))
                                  for family in settings["families"]},
            "primary_by_loading": {loading: summarize(data, METHODS, primary & (data["loading"] == loading))
                                   for loading in LOADINGS},
            "dataset_provenance": {name: manifest["provenance"] for name, manifest in frozen["splits"][0].items()
                                   if name != "seed" and manifest is not None},
            "paired_primary_comparisons": comparisons, "robustness_comparisons": robustness,
            "by_conditioning": conditioning, "strongest_test_classical_descriptive": strongest,
            "baseline_selected_on_validation": frozen["classical_baseline_by_loading"],
            "success_gates": gate, "quantum_contribution_supported": all(gate.values()),
            "conclusion": ("Evidence of a finite-budget quantum sampler contribution on this simulated benchmark."
                           if all(gate.values()) else
                           "Quantum contribution target NOT demonstrated; do not claim a benchmark win or quantum advantage."),
            "cost": {"training": frozen["cost"], "test_by_method": counters,
                     "training_seconds": frozen["training_seconds"], "test_wall_seconds": wall_seconds,
                     "first_test_detector_invocation_s": first_invocation_s,
                      "fresh_process_cold_probe": cold_probe,
                     "warmup_cost": warmup_cost,
                     "hardware_qpu_calls": 0, "hardware_qpu_seconds": 0,
                     "hardware_cost": "not applicable; local exact simulation only"},
            "protocol": {"quantum_runtime": "finite shots only; no optimum enumeration or parameter optimization",
                         "matching": "same k/shots/rounds/top-candidate/refinement budgets; SA work and all timings separate",
                         "reference": "exhaustive ML offline only, with the same receiver CSI",
                         "residual": "non-increasing against the classical incumbent; no BER guarantee",
                         "ablations": "perfect CSI only; warm/standard retrained, other switches use frozen angles",
                         "legacy": "mMobile axis-remapping results are exploratory and excluded",
                         "primary_scope": "pooled declared families; per-family cells retained",
                         "conditioning": "condition number of the vertically stacked real/imag receiver CSI",
                         "qpu_claim": "none; simulator latency is not 5G radio real-time latency",
                         "timing": "warm calls exclude training; worker count records CPU contention; cold probe separate"},
            "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                            "platform": platform.platform()}}


_WORKER_STATE = None


def _worker_init(banks, frozen, settings, ablations):
    global _WORKER_STATE
    _WORKER_STATE = banks, frozen, settings, ablations


def _worker_cell(condition):
    banks, frozen, settings, ablations = _WORKER_STATE
    return evaluate_cell(banks, *condition, settings["uses_per_seed"], frozen, ablations=ablations)


def run_test(run_dir, banks, families, frozen, settings, *, resume=False, ablations=True):
    verify_freeze(frozen)
    marker = run_dir / "test_started.json"
    identity = {"freeze_sha256": frozen["freeze_sha256"], "settings": settings, "ablations": ablations}
    if marker.exists():
        if not resume or json.loads(marker.read_text()) != identity:
            raise ValueError("Test already started; only identical --resume is permitted, without retuning")
    else:
        save_json(marker, identity)
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    cell_reports, collected, replays = [], [], []
    counters = {name: {key: 0 for key in ("shots", "circuit_invocations", "candidates", "candidate_refinements",
                                         "sa_spin_proposals", "fallbacks", "residual_violations")} for name in METHODS}
    # Separate first invocation from warm per-channel timing; not used for training or scoring.
    cold_instance = banks[0].instance("validation", families[0], "square", 12, 999)
    warmup, first_seconds = timed(detect, cold_instance.H, cold_instance.y, cold_instance.sigma2,
                                  config=Config(**frozen["config"]), params=np.asarray(frozen["params"]))
    warmup_cost = {key: warmup.telemetry[key] for key in ("shots", "circuit_invocations", "candidates")}
    cold_started = time.perf_counter()
    cold_result = subprocess.run([sys.executable, "-B", "-m", "signal_rescue.probe", str((run_dir / "frozen.json").resolve())],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    cold_probe = {**json.loads(cold_result.stdout), "process_wall_s": time.perf_counter() - cold_started,
                  "includes": "interpreter, imports, artifact read, channel creation, one detection"}
    conditions = [(family, loading, snr, csi) for family in families for loading in LOADINGS
                  for snr in SNRS for csi in CSI_LEVELS]
    pending = [condition for condition in conditions if not (resume and
               (cells_dir / ("_".join(map(str, condition)) + ".npz")).exists() and
               (cells_dir / ("_".join(map(str, condition)) + ".json")).exists())]
    executor = None
    if settings["workers"] > 1 and pending:
        executor = ProcessPoolExecutor(max_workers=settings["workers"], mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_worker_init, initargs=(banks, frozen, settings, ablations))
        calculated = iter(executor.map(_worker_cell, pending, chunksize=1))
    else:
        calculated = iter(evaluate_cell(banks, *condition, settings["uses_per_seed"], frozen, ablations=ablations)
                          for condition in pending)
    try:
        for family, loading, snr, csi in conditions:
            stem = f"{family}_{loading}_{snr}_{csi}"
            path, metadata_path = cells_dir / f"{stem}.npz", cells_dir / f"{stem}.json"
            if resume and path.exists() and metadata_path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    arrays = dict(saved)
                metadata = json.loads(metadata_path.read_text())
                costs, replay = metadata["cost"], metadata["replay"]
            else:
                arrays, costs, replay = next(calculated)
                np.savez_compressed(path, **arrays)
                save_json(metadata_path, {"cost": costs, "replay": replay})
            n = len(arrays["bits"])
            summary = summarize(arrays, METHODS)
            baseline = frozen["classical_baseline_by_loading"][loading]
            comparison = paired_bootstrap(arrays["errors"], arrays["bits"], arrays["groups"],
                                          METHODS.index("qaoa"), METHODS.index(baseline), resamples=1000)
            cell_reports.append({"family": family, "loading": loading, "snr_db": snr, "csi_error_db": csi,
                                 "channel_uses": n, "summary": summary, "paired_vs_validation_baseline": comparison})
            for name in METHODS:
                for key in counters[name]:
                    counters[name][key] += costs[name][key]
            collected.append({**arrays, "family": np.full(n, family), "loading": np.full(n, loading),
                              "snr": np.full(n, snr), "csi": np.full(n, csi)})
            if replay:
                replays.append(replay)
            print(f"  test {family}/{loading} SNR={snr:2} CSI={csi:>7}: "
                  f"BER qaoa={summary['qaoa']['ber']:.5f} {baseline}={summary[baseline]['ber']:.5f} "
                  f"n={n}", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    data = {key: np.concatenate([cell[key] for cell in collected]) for key in collected[0]}
    report = build_report(data, cell_reports, frozen, settings, counters, time.perf_counter() - started,
                          first_seconds, cold_probe, warmup_cost)
    np.savez_compressed(run_dir / "observations.npz", **data)
    save_json(run_dir / "report.json", report)
    save_json(run_dir / "replays.json", replays)
    from signal_rescue.presentation import make_artifacts
    make_artifacts(run_dir, report, replays)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--dataset", choices=("controlled", "deepmimo", "both"), default="both")
    parser.add_argument("--dataset-path", type=Path, default=ROOT / "data/deepmimo/asu_slice.npz")
    parser.add_argument("--output", type=Path, help="New versioned result directory (never old report files)")
    parser.add_argument("--stage", choices=("all", "train", "test"), default="all")
    parser.add_argument("--frozen", type=Path, help="Use an existing immutable training artifact")
    parser.add_argument("--resume", action="store_true", help="Continue completed cells of the identical frozen test")
    parser.add_argument("--uses-per-seed", type=int)
    parser.add_argument("--configs", type=int)
    parser.add_argument("--max-evals", type=int)
    parser.add_argument("--starts", type=int)
    parser.add_argument("--workers", type=int, help="1..4 independent CPU workers; recorded in latency protocol")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args(argv)
    settings = dict(PROFILES[args.profile])
    for key in ("uses_per_seed", "configs", "max_evals", "starts", "seeds", "workers"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if (settings["uses_per_seed"] < 1 or len(set(settings["seeds"])) != len(settings["seeds"])
            or not 1 <= settings["workers"] <= 4):
        parser.error("Require positive channel uses, unique seeds, and 1..4 workers")
    dataset = args.dataset
    status = {"DeepMIMO": "available", "source_cap_bytes": 100_000_000, "processed_cap_bytes": 20_000_000}
    if dataset == "both" and not args.dataset_path.exists():
        dataset, status["DeepMIMO"] = "controlled", "unavailable: no bounded local slice; controlled models only"
    families = list(FAMILIES) if dataset in ("controlled", "both") else []
    if dataset in ("deepmimo", "both"):
        families.append("deepmimo")
    banks = [ExperimentBank(seed, dataset=dataset, path=args.dataset_path) for seed in settings["seeds"]]
    settings.update(profile=args.profile, dataset=dataset, dataset_status=status, families=families,
                    detector="MMSE + bounded single/pair refinement + coupled block sampler",
                    candidate_samplers=["qaoa", *MATCHED], budget="frozen config; all sampler/refinement budgets matched")
    settings["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT / "signal_rescue").glob("*.py"))}
    run_dir = args.output or ROOT / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        parser.error("--resume requires an existing --output directory")
    print(f"Signal Rescue v2 / {settings['profile']} / {dataset}: {run_dir}", flush=True)
    if args.frozen or args.resume:
        frozen_path = args.frozen or run_dir / "frozen.json"
        frozen = json.loads(frozen_path.read_text())
        verify_freeze(frozen)
        if frozen["splits"] != [bank.manifest() for bank in banks]:
            raise ValueError("Frozen dataset/split provenance differs from this run")
        if not (run_dir / "frozen.json").exists():
            save_json(run_dir / "frozen.json", frozen)
    else:
        frozen = train_and_freeze(banks, families, configs=settings["configs"], max_evals=settings["max_evals"],
                                  starts=settings["starts"], training_uses=settings["training_uses"],
                                  validation_uses=settings["validation_uses"], seed=settings["seeds"][0])
        save_json(run_dir / "frozen.json", frozen)
    if args.resume and json.loads((run_dir / "settings.json").read_text()) != settings:
        raise ValueError("Resume settings or source code differ from the original frozen test")
    save_json(run_dir / "settings.json", settings)
    if args.stage != "train":
        report = run_test(run_dir, banks, families, frozen, settings, resume=args.resume,
                          ablations=not args.no_ablations)
        print(report["conclusion"], flush=True)
        print(f"Report and local demo: {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
