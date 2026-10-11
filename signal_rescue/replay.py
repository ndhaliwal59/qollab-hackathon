"""Bounded single-channel diagnostic runner; never updates confirmation scores."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from signal_rescue.benchmark import verify_freeze
from signal_rescue.channels import CSI_LEVELS, FAMILIES, LOADINGS, ExperimentBank
from signal_rescue.core import cost, mmse, ordered_sic
from signal_rescue.hybrid import detect, prepare
from signal_rescue.samplers import Config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--dataset", choices=("controlled", "deepmimo"), default="controlled")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/deepmimo/asu_slice.npz"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--detector", choices=("hybrid", "mmse", "mmse_sic", "classical"), default="hybrid")
    parser.add_argument("--sampler", choices=("qaoa", "uniform", "biased", "sa"), default="qaoa")
    parser.add_argument("--family", choices=(*FAMILIES, "deepmimo"), default="rayleigh")
    parser.add_argument("--loading", choices=LOADINGS, default="overloaded")
    parser.add_argument("--snr", type=float, default=12)
    parser.add_argument("--csi", choices=CSI_LEVELS, default="perfect")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--shots", type=int, choices=(64, 128))
    parser.add_argument("--block-size", type=int, choices=(4, 6, 8))
    parser.add_argument("--rounds", type=int, choices=(1, 2))
    parser.add_argument("--refine-steps", type=int)
    args = parser.parse_args(argv)
    frozen = json.loads(args.frozen.read_text())
    verify_freeze(frozen)
    config = Config(**frozen["config"])
    overrides = {key: value for key, value in (("shots", args.shots), ("k", args.block_size),
                 ("rounds", args.rounds), ("refine_steps", args.refine_steps)) if value is not None}
    config = replace(config, **overrides)
    family = "deepmimo" if args.dataset == "deepmimo" else args.family
    bank = ExperimentBank(args.seed, dataset=args.dataset, path=args.dataset_path)
    instance = bank.instance(args.split, family, args.loading, args.snr, args.index, csi=args.csi)
    started = time.perf_counter()
    detail, telemetry = None, None
    if args.detector == "hybrid":
        result = detect(instance.H, instance.y, instance.sigma2, kind=args.sampler, config=config,
                        params=np.asarray(frozen["params"]), seed=args.seed, replay=True)
        x, detail, telemetry = result.x, result.replay, result.telemetry
    elif args.detector == "mmse":
        x = mmse(instance.H, instance.y, instance.sigma2)[0]
    elif args.detector == "mmse_sic":
        x = ordered_sic(instance.H, instance.y, instance.sigma2)
    else:
        x = prepare(instance.H, instance.y, instance.sigma2, config)[-1]
    print(json.dumps({"purpose": "single-channel diagnostic only, not a new confirmation test",
                      "config": config.dictionary(), "budget_overrides": overrides,
                      "group": instance.group, "truth": instance.truth.tolist(), "decision": x.tolist(),
                      "bit_errors": int(np.count_nonzero(x != instance.truth)),
                      "residual": cost(instance.H, instance.y, x), "seconds": time.perf_counter() - started,
                      "telemetry": telemetry, "replay": detail}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
