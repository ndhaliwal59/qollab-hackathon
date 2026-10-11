"""Bounded shared-parameter training, validation selection, then immutable freeze."""

from __future__ import annotations
from dataclasses import replace
import hashlib
import itertools
import json
import time
import numpy as np
from scipy.optimize import minimize
from signal_rescue.channels import LOADINGS, PRIMARY_SNRS, keyed_seed
from signal_rescue.core import ising, mmse, select_block, soft_probabilities
from signal_rescue.hybrid import detect, prepare
from signal_rescue.samplers import Config, objective_value, probabilities, spectrum_for_simulation


def candidate_configs(limit=24):
    if not 1 <= limit <= 24:
        raise ValueError("Search is capped at 24 configurations")
    # Predeclared balanced fractional search, NOT exhaustive hyperparameter tuning.
    grid = []
    for warm in (False, True):
        for p in (1, 2, 3):
            for k in (4, 6, 8):
                i = len(grid)
                grid.append(Config(k=k, p=p, warm=warm, shots=(64, 128)[i % 2],
                                   epsilon=(.1, .25)[(i // 2) % 2],
                                   objective=("mean", "cvar")[(i + p) % 2],
                                   alpha=(.1, .25)[i % 2]))
    grid += [replace(grid[i], shots=192 - grid[i].shots,
                     objective="cvar", alpha=.25 if grid[i].alpha == .1 else .1)
             for i in (0, 4, 8, 9, 13, 17)]
    # Interleave warm/standard and depths so reduced smoke searches still cover both.
    order = [j for pair in zip(range(9), range(9, 18)) for j in pair] + list(range(18, 24))
    return [grid[i] for i in order[:limit]]


def examples(banks, split, families, count):
    result = []
    conditions = [(family, loading, snr) for snr, family, loading in
                  itertools.product(PRIMARY_SNRS, families, LOADINGS)]
    for index in range(count):
        family, loading, snr = conditions[index % len(conditions)]
        channel_index = index // (len(families) * len(LOADINGS))
        bank = banks[channel_index % len(banks)]
        instance = bank.instance(split, family, loading, snr, channel_index // len(banks))
        result.append((loading, instance))
    return result


def calibrate(examples_):
    softs, posteriors, labels = [], [], []
    for _, instance in examples_:
        _, soft, posterior = mmse(instance.H, instance.y, instance.sigma2)
        softs.extend(soft)
        posteriors.extend(posterior)
        labels.extend(instance.truth < 0)
    soft, posterior, labels = np.asarray(softs), np.asarray(posteriors), np.asarray(labels)
    temperatures = [.25, .5, 1., 2., 4., 8., 16.]
    losses = []
    for temperature in temperatures:
        q = soft_probabilities(soft, posterior, temperature=temperature, epsilon=.001)
        losses.append(float(-np.mean(labels * np.log(q) + (1 - labels) * np.log1p(-q))))
    return temperatures[int(np.argmin(losses))], dict(zip(map(str, temperatures), losses))


def prepare_training(examples_, config):
    prepared = []
    for _, instance in examples_:
        G, b, _, soft, posterior, incumbent = prepare(instance.H, instance.y, instance.sigma2, config)
        idx = select_block(soft, posterior, G, config.k, coupled=config.coupled)
        h, J = ising(G, b, incumbent, idx)
        q = soft_probabilities(soft[idx], posterior[idx], temperature=config.temperature,
                               epsilon=config.epsilon)
        prepared.append((h, J, q, spectrum_for_simulation(h, J)))
    return prepared


def fit_parameters(prepared, config, *, max_evals=50, starts=3, seed=7,
                   finite=False, initial=None):
    if not 2 * config.p + 2 <= max_evals <= 50 or not 1 <= starts <= 3:
        raise ValueError("Optimizer needs at least 2*p+2 evaluations and is capped at 3 starts / 50 evaluations")
    rng = np.random.default_rng(seed)
    seeds = [rng.integers(0, 2**32, size=len(prepared)) for _ in range(max_evals)]
    counts = {"objective_evaluations": 0, "circuit_invocations": 0, "shots": 0,
              "statevector_evaluations": 0, "stage": "finite_shot" if finite else "exact_statevector"}
    best_value, best_params = np.inf, None
    for start in range(starts):
        point = np.concatenate((np.full(config.p, .3), np.full(config.p, .4)))
        if start:
            point += rng.normal(scale=.4, size=len(point))
        elif initial is not None:
            old_p = len(initial) // 2
            # Layerwise initialization from the previous trained depth.
            point[:min(old_p, config.p)] = initial[:min(old_p, config.p)]
            point[config.p:config.p + min(old_p, config.p)] = initial[old_p:old_p + min(old_p, config.p)]
        calls = 0

        def objective(theta):
            nonlocal calls
            values = []
            for j, (h, J, q, diagonal) in enumerate(prepared):
                probs = probabilities(h, J, theta, p=config.p, warm_prob=q if config.warm else None)
                counts["statevector_evaluations"] += 1
                counts["circuit_invocations"] += 1
                if finite:
                    # Common random numbers stabilize noisy local training, not an exact expectation.
                    shot_rng = np.random.default_rng(int(seeds[start % max_evals][j]))
                    sampled = shot_rng.choice(len(probs), config.shots, p=probs)
                    probs = np.bincount(sampled, minlength=len(probs)) / config.shots
                    counts["shots"] += config.shots
                values.append(objective_value(diagonal, probs, objective=config.objective,
                                              alpha=config.alpha))
            calls += 1
            counts["objective_evaluations"] += 1
            return float(np.mean(values))

        result = minimize(objective, point, method="COBYLA",
                          options={"maxiter": max_evals, "rhobeg": .3, "tol": 1e-3})
        if result.fun < best_value:
            best_value, best_params = float(result.fun), result.x.copy()
    return best_params, {**counts, "best_training_objective": best_value}


def validation_score(examples_, config, params, *, kind="qaoa", seed=7):
    errors = bits = 0
    by_loading = {name: [0, 0] for name in LOADINGS}
    cost = {"shots": 0, "circuit_invocations": 0, "candidates": 0}
    for i, (loading, instance) in enumerate(examples_):
        if kind == "classical":
            x = prepare(instance.H, instance.y, instance.sigma2, config)[-1]
            work = {key: 0 for key in cost}
        else:
            result = detect(instance.H, instance.y, instance.sigma2, config=config, params=params,
                            kind=kind, seed=keyed_seed(seed, "validation-sampler", i))
            x, work = result.x, result.telemetry
        wrong = int(np.count_nonzero(x != instance.truth))
        n = len(instance.truth)
        errors += wrong
        bits += n
        by_loading[loading][0] += wrong
        by_loading[loading][1] += n
        for key in cost:
            cost[key] += work[key]
    return {"ber": errors / bits, "bit_errors": errors, "bits": bits,
            "by_loading": {key: (e / n if n else None) for key, (e, n) in by_loading.items()},
            "cost": cost}


def train_and_freeze(banks, families, *, configs=24, max_evals=50, starts=3,
                     training_uses=48, validation_uses=240, seed=7):
    started = time.perf_counter()
    training = examples(banks, "train", families, training_uses)
    validation = examples(banks, "validation", families, validation_uses)
    temperature, calibration = calibrate(training)
    trials, depth_params = [], {}
    total_cost = {"objective_evaluations": 0, "circuit_invocations": 0, "shots": 0,
                  "statevector_evaluations": 0, "validation_shots": 0,
                  "validation_circuit_invocations": 0}

    def add_cost(cost):
        for key in ("objective_evaluations", "circuit_invocations", "shots", "statevector_evaluations"):
            total_cost[key] += cost[key]

    def validate_and_count(config, params, kind="qaoa"):
        score = validation_score(validation, config, params, kind=kind, seed=seed)
        total_cost["validation_shots"] += score["cost"]["shots"]
        total_cost["validation_circuit_invocations"] += score["cost"]["circuit_invocations"]
        return score

    for index, candidate in enumerate(candidate_configs(configs)):
        config = replace(candidate, temperature=temperature)
        prepared = prepare_training(training, config)
        initial = depth_params.get((config.k, config.warm, config.p - 1))
        params, cost = fit_parameters(prepared, config, max_evals=max_evals, starts=starts,
                                      seed=seed + index, initial=initial)
        depth_params[(config.k, config.warm, config.p)] = params
        add_cost(cost)
        score = validate_and_count(config, params)
        trials.append({"config": config.dictionary(), "params": params.tolist(),
                       "training": cost, "validation": score})
        print(f"  train {index + 1}/{configs}: k={config.k} p={config.p} warm={config.warm} "
              f"{config.objective} shots={config.shots}, validation BER={score['ber']:.5f}", flush=True)

    # Finite-shot retrain two validation finalists, within a separately recorded cap.
    finalists = sorted(range(len(trials)), key=lambda i: trials[i]["validation"]["ber"])[:2]
    for index in finalists:
        trial = trials[index]
        config = Config(**trial["config"])
        params, cost = fit_parameters(prepare_training(training, config), config,
                                      max_evals=max_evals, starts=starts, seed=seed + 10_000 + index,
                                      finite=True, initial=np.asarray(trial["params"]))
        add_cost(cost)
        trial["finite_shot_training"] = cost
        trial["params"] = params.tolist()
        trial["validation"] = validate_and_count(config, params)
    # Selection is restricted to the finite-shot retrained finalists.
    selected = min((trials[i] for i in finalists), key=lambda t: t["validation"]["ber"])
    config, params = Config(**selected["config"]), np.asarray(selected["params"])
    classical_scores = {kind: validate_and_count(config, params, kind)
                        for kind in ("classical", "uniform", "biased", "sa")}
    baseline_by_loading = {loading: min(classical_scores,
                                       key=lambda kind: classical_scores[kind]["by_loading"][loading])
                           for loading in LOADINGS}

    # Single-factor warm/standard ablation, same k/p/shots and training budget.
    opposite = replace(config, warm=not config.warm)
    opposite_params, opposite_cost = fit_parameters(prepare_training(training, opposite), opposite,
                                                    max_evals=max_evals, starts=starts,
                                                    seed=seed + 20_000, finite=True, initial=params)
    add_cost(opposite_cost)
    opposite_score = validate_and_count(opposite, opposite_params)
    frozen = {"schema_version": 2, "config": config.dictionary(), "params": params.tolist(),
              "opposite_warm_config": opposite.dictionary(), "opposite_warm_params": opposite_params.tolist(),
              "opposite_warm_validation": opposite_score, "opposite_warm_training": opposite_cost,
              "classical_baseline_by_loading": baseline_by_loading,
              "classical_validation": classical_scores, "search_trials": trials,
              "training_uses": training_uses, "validation_uses": validation_uses,
              "calibration_temperature": temperature, "calibration_training_log_loss": calibration,
              "cost": total_cost, "training_seconds": time.perf_counter() - started,
              "splits": [bank.manifest() for bank in banks],
              "primary_snrs_db": PRIMARY_SNRS,
              "search_caps": {"configurations": 24, "starts_per_stage": 3,
                              "objective_evaluations_per_start_per_stage": 50,
                              "finite_shot_finalists": 2},
              "selection": "training optimization + validation selection only; test never consulted"}
    frozen["freeze_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    return frozen
