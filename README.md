# Signal Rescue v2

5G-focused, **complex BPSK multi-user detection** with a bounded quantum/classical hybrid.
This is a channel-detection experiment, not a standards-compliant 5G NR modem, decoded
network traffic, or a demonstration of hardware quantum advantage.

## Run

### Live interactive laboratory

The live local application uses the existing Python environment and executes real
Qiskit circuits with `BasicSimulator`. From the repository root:

```sh
.venv/bin/python -m signal_rescue.local_app
```

It opens `http://127.0.0.1:8000`. Use `--no-browser` to suppress the browser tab or
`--port 8001` to choose another port. The server binds to localhost by default; no
hosted backend, credentials, or external API is needed. Classical previews execute no
circuits. The explicit run executes two adaptive circuits and exports its measured
counts and detector results as JSON.

To create the platform handoff package, run:

```sh
.venv/bin/python scripts/package_qollab.py
```

The resulting `dist/qollab/` contains only the Qollab entry files, the shared Python
service and required modules, the small validation channel slice, and frozen benchmark
artifacts. See [the demo guide](docs/DEMO.md) and [Qollab handoff](docs/QOLLAB_HANDOFF.md).

The archived `results/*/demo.html` files remain precomputed historical replays. They
are separate from this live application and are not used as its quantum measurements.

Use the existing Python environment, or install `requirements.txt` in a virtual environment.
Only NumPy/SciPy/Matplotlib are needed by the new runner; Qiskit validates the circuit kernel.
`requirements-confirmation.txt` captures the exact installed package versions used for this run.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m signal_rescue.benchmark --profile smoke
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m signal_rescue.benchmark --profile full
```

`python -m signal_rescue.run_benchmark` uses the same v2 5G benchmark.
This repository is only Signal Rescue:

```
signal_rescue/   detectors, QAOA kernel, training, and the benchmark
data/            DeepMIMO slice and the deprecated mMobile mats
results/         confirmation-full, plus smoke and preflight checks
tests/
```

Every run uses a new directory under `results/`. Open its `demo.html`
directly in a browser; it loads precomputed results and a bounded replay, with no server,
network calls, dependency bundle, or large dataset loaded by the browser.

Full profile: 5 seeds × 200 uses per seed × 4 channel families × 2 loadings × 6 SNRs ×
3 CSI conditions = **144,000 scored channel uses**, with 1,000 uses per condition cell.
Some are repeated symbols/noise on the same channel; they are NOT 144,000 independent
physical channels. The paired bootstrap clusters these repeats appropriately.
Four independent CPU worker processes bound runtime; `--workers 1` gives uncontended
single-process timings. Worker count is recorded. BLAS threads are limited in the
commands above to prevent CPU oversubscription.

## Small real-channel dataset

The included DeepMIMO ASU campus source is **ray traced at 3.5 GHz**, not measured
over-the-air signals. Its 30,558,838-byte archive is below the user-approved **100 MB
source/download cap**. The optional 94,990,528-byte interaction member also fits this
cap, but is not extracted: detection only needs powers, phases, departure angles,
delays, and user positions. The browser-facing/processed cap remains **20 MB**.

The processed slice is about **103 KB**, with 126 8×12 complex channel matrices in
42 spatial groups. Each matrix combines 12 distinct single-antenna users at the SAME
subcarrier and one 8-antenna BS. The square case takes the first 8 users. A single
Frobenius scalar sets received SNR; column powers and complex phase are retained.

Recreate from the official archive (existing files are never overwritten):

```sh
.venv/bin/python -B -m signal_rescue.dataset --scenario asu_campus_3p5 --output data/deepmimo/asu_campus_3p5.zip
.venv/bin/python -B -m signal_rescue.dataset --convert data/deepmimo/asu_campus_3p5.zip --slice-output data/deepmimo/asu_slice.npz
```

The converter specializes [DeepMIMO's channel equations](https://github.com/DeepMIMO/DeepMIMO/blob/82be994e53a2e37d88a3575a3f6fd35bb6611b53/deepmimo/generator/channel.py)
and [array geometry](https://github.com/DeepMIMO/DeepMIMO/blob/82be994e53a2e37d88a3575a3f6fd35bb6611b53/deepmimo/generator/geometry.py)
to an isotropic half-wavelength `[8,1]` ULA, one UE antenna, 512 OFDM bins, 10 MHz
bandwidth, no receive LPF, no Doppler, and bin 0. It sums the stored ray phases and
path powers; reciprocal uplink uses a transpose, not a Hermitian transpose.

Spatial tiles are 64 m wide with 8 m boundary guards. Disjoint user sets and guard
gaps keep adjacent retained positions within the same spatial group. Whole groups
are split 60/20/20 BEFORE symbols/noise. DeepMIMO's physical spatial split is fixed
across all noise seeds, and its bootstrap groups are also shared across seeds.
Only nine spatial tiles are held out: uncertainty on this small slice can be wide.
Generalization to unseen sites is NOT established by this single-campus dataset.

`--dataset controlled|deepmimo|both` and `--dataset-path` select inputs. If `both`
cannot find a local slice, the report explicitly marks DeepMIMO unavailable and runs
controlled models only. Explicit `deepmimo` fails if missing/malformed/oversized.
NPZ data is loaded without pickle, expanded size is checked, and provenance is required.

The old mMobile adapter remapped beam/subcarrier axes to antenna axes and discarded
phase. Its reports remain **deprecated exploratory artifacts**, excluded from scoring.
Reproducing that old runner requires an explicit `exploratory=True` argument.

## Detector and fairness

`MMSE → bounded single/pair search → uncertain coupled-bit block → finite-shot sampler
→ common bounded refinement → accept only a lower-residual result`.

For BPSK, `G = Re(HᴴH)`, `b = Re(Hᴴy)`. Circular complex noise has `E|n|² = sigma2`,
so each real component has variance `sigma2/2`; the equivalent real-BPSK MMSE uses
`G + sigma2/2 I`. Fixed bits enter the free-block Ising linear field exactly.

Blocks have 4/6/8 bits, selected from MMSE posterior variance, soft margins, and
normalized Gram correlations. A second round seeds a previously excluded bit.
Warm-start probabilities are calibrated on TRAINING labels only and clipped at
epsilon 0.1 or 0.25. Two correction rounds, 64/128 candidates per round, four distinct
sampled finalists (incumbent padding if fewer), and at most eight single/pair steps
per finalist are shared with uniform, independent-biased, and simulated-annealing
samplers. Annealing's extra spin proposals and latency are reported separately.

The incumbent survives any sampler failure. Residual cannot increase against that
incumbent, but **lower residual does not guarantee lower transmitted-bit error**.
Exact ML enumerates at most 12 bits OFFLINE, after the hybrids have already run.
The quantum runtime never calls an optimum oracle or an optimizer. It evolves a
small exact statevector and draws only its finite shot budget; classical simulation
of all amplitudes is NOT itself a quantum speedup.

Baselines: MMSE, ordered MMSE-SIC, single-bit search, single/pair search, three matched
classical-sampler hybrids, and exhaustive ML. Ablations: standard vs warm QAOA
(same architecture and retrained), coupling vs uncertainty only, refinement vs none,
and complex vs real-only channels. Other ablations retain frozen angles rather than
receiving additional test-informed tuning. Ablations run with perfect CSI only.

## Training and confirmation

The fixed fractional search covers depths 1/2/3, blocks 4/6/8, 64/128 shots,
standard/warm QAOA, mean energy, and CVaR alpha 0.1/0.25. It is not an exhaustive
factorial search. Maximum: 24 search configs, three optimizer starts, 50 objective
evaluations per start per stage. Shared angles train across channel batches, using
layerwise initialization when a prior depth is available. Two validation finalists
are retrained with finite shots. A separately reported warm/standard ablation fit
does not enter winner selection. All simulation invocations, evaluations, and shots
are counted; there are **zero QPU/hardware calls**.

`frozen.json` (SHA-256 integrity checked) is written before any confirmation test.
Validation selects the quantum config and the classical baseline per loading.
No failed test triggers retuning. Primary SNRs are fixed at **8/12/16/20 dB**, not
selected by observed performance. The full sweep also includes 0/4 dB, Rayleigh,
correlated Rayleigh, Rician, and DeepMIMO, and CSI errors perfect/−20/−10 dB.

The contribution gate requires ≥5% relative BER improvement over the strongest
classical hybrid (including classical-only refinement), a positive paired group-bootstrap
confidence interval against EVERY classical hybrid, no supported aggregate regression per loading/CSI,
adequate sample size, no sampler failures, and warm local simulator p95 <1 second.
Passing simulator latency is not a real-time 5G latency claim. Confidence intervals
are descriptive, unadjusted 95% intervals; individual condition findings are exploratory.

Artifacts: `report.json`, `frozen.json`, `settings.json`, compressed paired
`observations.npz`, per-condition checkpoints, `replays.json`, three PNG plots,
and `demo.html`. Report BER, BLER, residual, p95/mean latency, grouped paired CIs,
per-family/loading/SNR/CSI/conditioning breakdowns, source hashes, and separate cold,
training, validation, and test work.

An interrupted identical test can continue WITHOUT retuning:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m signal_rescue.benchmark --profile full --output results/RUN_NAME --resume
```

Resume rejects changed source/settings/freeze. Completed cells are reused; old
reports are not overwritten by new runs. `--stage train` saves a model without
testing; `--stage test --frozen PATH` evaluates a previously trained artifact.

## Single-channel diagnostic runner

This exposes dataset, split, detector, sampler, seed, and bounded-work options without
changing the fixed confirmation protocol or its saved scores:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m signal_rescue.replay --frozen results/RUN_NAME/frozen.json --dataset deepmimo --split validation --detector hybrid --sampler qaoa --loading overloaded --snr 12 --seed 7 --index 0 --shots 64 --block-size 6 --rounds 2 --refine-steps 8
```

Choose detector `hybrid|mmse|mmse_sic|classical`, sampler `qaoa|uniform|biased|sa`,
or controlled family `rayleigh|correlated|rician`. Budget overrides are labeled
diagnostic only; they do not turn into new confirmation results.
