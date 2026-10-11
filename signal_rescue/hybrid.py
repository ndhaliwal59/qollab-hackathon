"""MMSE -> bounded classical search -> coupled correction -> common refinement."""

from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from signal_rescue.core import (cost, energies, ising, mmse, quadratic, refine,
                                select_block, soft_probabilities)
from signal_rescue.samplers import Config, sample


@dataclass
class Detection:
    x: np.ndarray
    telemetry: dict
    replay: dict | None = None


def prepare(H, y, sigma2, config: Config):
    hard, soft, posterior = mmse(H, y, sigma2)
    G, b = quadratic(H, y)
    incumbent = refine(G, b, hard, steps=config.refine_steps)[0][0]
    return G, b, hard, soft, posterior, incumbent


def detect(H, y, sigma2, *, kind="qaoa", config=None, params=None, seed=0,
           prepared=None, replay=False, sampler_fn=sample) -> Detection:
    config = config or Config()
    started = time.perf_counter()
    G, b, hard, soft, posterior, incumbent = (prepare(H, y, sigma2, config)
                                             if prepared is None else prepared)
    incumbent = incumbent.copy()
    initial = incumbent.copy()
    current_cost = cost(H, y, incumbent)
    baseline_cost = current_cost
    q = soft_probabilities(soft, posterior, temperature=config.temperature, epsilon=config.epsilon)
    telemetry = {"shots": 0, "circuit_invocations": 0, "candidates": 0,
                 "candidate_refinements": 0, "accepted_rounds": 0, "fallbacks": 0,
                 "sa_spin_proposals": 0, "sampler_seconds": 0., "refinement_seconds": 0.,
                 "rounds": [], "residual_before": current_cost}
    previous = ()
    replay_rounds = []
    for round_index in range(config.rounds):
        idx = select_block(soft, posterior, G, config.k, coupled=config.coupled,
                           round_index=round_index, previous=previous)
        previous = idx
        h, J = ising(G, b, incumbent, idx)
        incumbent_before = incumbent.copy()
        try:
            sampled = sampler_fn(kind, h, J, q[idx], config, params, seed + 7919 * round_index)
            candidates = sampled.candidates
            if (candidates.shape != (config.shots, len(idx)) or not np.all(np.isfinite(candidates))
                    or np.any(np.abs(candidates) != 1)):
                raise ValueError("Sampler returned malformed candidates")
            # Rank SAMPLED candidates only. The exact spectrum is not queried.
            unique = np.unique(candidates, axis=0)
            order = np.argsort(energies(unique, h, J), kind="stable")[:config.finalists]
            expanded = np.tile(incumbent, (config.finalists, 1))
            expanded[:len(order), idx] = unique[order]
            refine_started = time.perf_counter()
            refined, moves = refine(G, b, expanded, steps=config.refine_steps)
            telemetry["refinement_seconds"] += time.perf_counter() - refine_started
            values = np.einsum("bi,ij,bj->b", refined, G, refined) - 2 * (refined @ b)
            trial = refined[int(np.argmin(values))]
            trial_cost = cost(H, y, trial)
            accepted = trial_cost < current_cost - 1e-10
            before = current_cost
            if accepted:
                incumbent, current_cost = trial.copy(), trial_cost
                telemetry["accepted_rounds"] += 1
            for key in ("shots", "circuit_invocations", "candidates", "sa_spin_proposals"):
                telemetry[key] += sampled.telemetry[key]
            telemetry["sampler_seconds"] += sampled.telemetry["seconds"]
            telemetry["candidate_refinements"] += config.finalists
            telemetry["rounds"].append({**sampled.telemetry, "bits": idx.tolist(),
                                         "accepted": bool(accepted), "moves": moves,
                                         "residual_before": before, "residual_after": current_cost})
            if replay:
                replay_rounds.append({"bits": idx.tolist(), "sampled_candidates": unique[order].tolist(),
                                      "incumbent_before": incumbent_before.tolist(),
                                      "incumbent_after": incumbent.tolist(),
                                      "sampled_energies": energies(unique[order], h, J).tolist(),
                                      "refined_candidates": refined.tolist(),
                                      "refined_residuals": [cost(H, y, candidate) for candidate in refined],
                                      "accepted": bool(accepted),
                                      "residual_before": before, "residual_after": current_cost})
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            # Failure cannot worsen the incumbent. Programming errors are NOT swallowed.
            telemetry["fallbacks"] += 1
            telemetry["rounds"].append({"bits": idx.tolist(), "fallback": str(error),
                                         "residual_after": current_cost})
    telemetry.update(seconds=time.perf_counter() - started, residual_after=current_cost,
                     residual_nonincreasing=bool(current_cost <= baseline_cost + 1e-8))
    detail = None
    if replay:
        detail = {"mmse_bits": hard.tolist(), "classical_incumbent": initial.tolist(),
                  "rounds": replay_rounds, "final_bits": incumbent.tolist(),
                  "note": "Lower residual is not a guarantee of lower transmitted-bit error."}
    return Detection(incumbent, telemetry, detail)
