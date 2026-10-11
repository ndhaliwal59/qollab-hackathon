"""Legacy exploratory runner. The CLI now routes to the corrected v2 benchmark."""

from __future__ import annotations

import time

import numpy as np

from signal_rescue.qaoa import scaled_expectation, solve_qaoa
from signal_rescue.detectors import (
    exact_ml,
    ising_coefficients,
    local_search,
    mmse_detect,
    qaoa_correct,
    random_correct,
    subproblem_optimum,
    uncertain_bits,
    zf_detect,
)
from signal_rescue.mmobile_data import SCENARIOS, SOURCE_METADATA, load_instance

SNRS_DB = [0, 4, 8, 12, 16, 20]
METHODS = ["zf", "mmse", "local_search", "random", "qaoa", "ml"]


def _errors(estimate: np.ndarray, truth: np.ndarray) -> int:
    decided = np.where(estimate < 0.0, -1.0, 1.0)
    return int(np.count_nonzero(decided != truth))


def _select_params(instances, *, k, p, shots, maxiter, n_starts, seed):
    """Optimize a few channels, then keep the parameter vector that transfers best."""
    prepared = []
    for offset, inst in enumerate(instances):
        hard, soft = mmse_detect(inst["H"], inst["y"], inst["sigma2"])
        idx = uncertain_bits(soft, k)
        h, J = ising_coefficients(inst["H"], inst["y"], hard, idx)
        result = solve_qaoa(
            h,
            J,
            p=p,
            shots=shots,
            seed=seed + offset,
            maxiter=maxiter,
            n_starts=n_starts,
        )
        prepared.append((h, J, result.params))
    best_params = prepared[0][2]
    best_score = np.inf
    for _h, _J, params in prepared:
        score = np.mean([scaled_expectation(h, J, params, p) for h, J, _params in prepared])
        if score < best_score:
            best_score = score
            best_params = params
    return best_params


def _focus_stats(ber: dict[str, list[float]], min_gap: float = 0.005):
    mmse = np.asarray(ber["mmse"], dtype=float)
    ml = np.asarray(ber["ml"], dtype=float)
    keep = (mmse - ml) > min_gap
    if not np.any(keep):
        keep = np.ones(len(mmse), dtype=bool)
    focus = {name: float(np.mean(np.asarray(values)[keep])) for name, values in ber.items()}
    denom = focus["mmse"] - focus["ml"]

    def closure(method: str) -> float:
        if denom <= 1e-12:
            return 0.0
        return float((focus["mmse"] - focus[method]) / denom)

    return keep, focus, closure


def run(
    *,
    n_realizations: int = 120,
    train: int = 4,
    shots: int = 64,
    p: int = 2,
    k: int = 8,
    maxiter: int = 25,
    n_starts: int = 2,
    seed: int = 7,
    snrs_db: list[float] | None = None,
    scenarios: dict | None = None,
    exploratory: bool = False,
) -> dict:
    if not exploratory:
        raise RuntimeError("Legacy mMobile axis remapping is not scored MIMO. Use python -m signal_rescue.benchmark; "
                           "explicit exploratory=True is required to reproduce the deprecated experiment.")
    snrs_db = list(SNRS_DB if snrs_db is None else snrs_db)
    scenarios = SCENARIOS if scenarios is None else scenarios
    report = {
        "data": SOURCE_METADATA["name"],
        "data_metadata": SOURCE_METADATA,
        "settings": {
            "n_realizations": n_realizations,
            "train": train,
            "shots": shots,
            "p": p,
            "k": k,
            "maxiter": maxiter,
            "n_starts": n_starts,
            "seed": seed,
            "snrs_db": snrs_db,
        },
        "scenarios": {},
    }
    for s_index, (name, shape) in enumerate(scenarios.items()):
        n_rx, n_tx = shape["n_rx"], shape["n_tx"]
        ber = {method: [] for method in METHODS}
        hit_qaoa = []
        hit_random = []
        prob_qaoa = []
        prob_random = []
        ml_match_qaoa = []
        ml_match_local = []
        subproblem_is_ml = []
        print(f"Signal Rescue / {name} ({n_tx}x{n_rx})", flush=True)
        for snr_index, snr in enumerate(snrs_db):
            instances = [
                load_instance(
                    n_rx,
                    n_tx,
                    snr,
                    np.random.default_rng(seed + s_index * 100_000 + snr_index * 1_000 + i),
                )
                for i in range(n_realizations)
            ]
            params = _select_params(
                instances[:train],
                k=k,
                p=p,
                shots=shots,
                maxiter=maxiter,
                n_starts=n_starts,
                seed=seed + 50_000 + s_index * 1_000 + snr_index,
            )
            totals = {method: 0 for method in METHODS}
            hits_q = hits_r = match_q = match_l = sub_ml = 0
            probs_q = []
            probs_r = []
            for i, inst in enumerate(instances):
                H, y, truth, sigma2 = inst["H"], inst["y"], inst["x"], inst["sigma2"]
                zf, _soft = zf_detect(H, y)
                mmse, _soft = mmse_detect(H, y, sigma2)
                searched = local_search(H, y, mmse)
                ml = exact_ml(H, y)
                sample_seed = seed + snr_index * 100 + i
                qaoa, qres, idx, h, J, hard = qaoa_correct(
                    H,
                    y,
                    sigma2,
                    k=k,
                    p=p,
                    shots=shots,
                    seed=sample_seed,
                    x0=params,
                    optimize=False,
                )
                random_x, rres, _idx, _h, _J = random_correct(
                    H, y, sigma2, k=k, shots=shots, seed=sample_seed + 50_000
                )
                totals["zf"] += _errors(zf, truth)
                totals["mmse"] += _errors(mmse, truth)
                totals["local_search"] += _errors(searched, truth)
                totals["random"] += _errors(random_x, truth)
                totals["qaoa"] += _errors(qaoa, truth)
                totals["ml"] += _errors(ml, truth)
                hits_q += int(qres.hit)
                hits_r += int(rres.hit)
                probs_q.append(qres.prob_optimal)
                probs_r.append(rres.prob_optimal)
                match_q += int(np.array_equal(np.where(qaoa < 0, -1.0, 1.0), ml))
                match_l += int(np.array_equal(searched, ml))
                optimal_subset = subproblem_optimum(h, J, hard, idx)
                sub_ml += int(np.array_equal(optimal_subset, ml))
            n_bits = n_realizations * n_tx
            for method in METHODS:
                ber[method].append(totals[method] / n_bits)
            hit_qaoa.append(hits_q / n_realizations)
            hit_random.append(hits_r / n_realizations)
            prob_qaoa.append(float(np.mean(probs_q)))
            prob_random.append(float(np.mean(probs_r)))
            ml_match_qaoa.append(match_q / n_realizations)
            ml_match_local.append(match_l / n_realizations)
            subproblem_is_ml.append(sub_ml / n_realizations)
            print(
                f"  SNR {snr:>5.1f} dB  BER mmse={ber['mmse'][-1]:.4f}  "
                f"local={ber['local_search'][-1]:.4f}  random={ber['random'][-1]:.4f}  "
                f"qaoa={ber['qaoa'][-1]:.4f}  ml={ber['ml'][-1]:.4f}",
                flush=True,
            )
        keep, focus, closure = _focus_stats(ber)
        report["scenarios"][name] = {
            "n_rx": n_rx,
            "n_tx": n_tx,
            "snrs_db": snrs_db,
            "ber": ber,
            "focus_snrs_db": [snr for snr, flag in zip(snrs_db, keep) if flag],
            "ber_focus": focus,
            "gap_closure": {method: closure(method) for method in ["local_search", "random", "qaoa"]},
            "ising_hit_rate_by_snr": {"qaoa": hit_qaoa, "random": hit_random},
            "ising_hit_rate": {
                "qaoa": float(np.mean(np.asarray(hit_qaoa)[keep])),
                "random": float(np.mean(np.asarray(hit_random)[keep])),
            },
            "prob_optimal": {
                "qaoa": float(np.mean(np.asarray(prob_qaoa)[keep])),
                "random": float(np.mean(np.asarray(prob_random)[keep])),
            },
            "ml_match_rate": {
                "qaoa": float(np.mean(np.asarray(ml_match_qaoa)[keep])),
                "local_search": float(np.mean(np.asarray(ml_match_local)[keep])),
            },
            "subproblem_is_global_ml_rate": float(np.mean(np.asarray(subproblem_is_ml)[keep])),
        }
    report["seconds_per_instance"] = _time_cold_start(p=p, shots=shots, maxiter=maxiter, n_starts=n_starts, seed=seed)
    print(f"Signal Rescue cold start: {report['seconds_per_instance']:.2f}s", flush=True)
    return report


def _time_cold_start(*, p, shots, maxiter, n_starts, seed) -> float:
    """One optimized 8-qubit detection instance, the Qollab-time proxy."""
    rng = np.random.default_rng(seed)
    inst = load_instance(8, 8, 8.0, rng)
    started = time.perf_counter()
    qaoa_correct(
        inst["H"],
        inst["y"],
        inst["sigma2"],
        k=8,
        p=p,
        shots=shots,
        seed=seed,
        maxiter=maxiter,
        n_starts=n_starts,
        optimize=True,
    )
    return time.perf_counter() - started


if __name__ == "__main__":
    from signal_rescue.benchmark import main
    main()
