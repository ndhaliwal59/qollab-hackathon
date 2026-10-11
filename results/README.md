# Benchmark result status

The current 5G implementation and confirmation are in
[confirmation-full](confirmation-full/).

The confirmation includes 144,000 channel uses, five seeds, complex BPSK channels,
a bounded DeepMIMO slice, matched classical samplers, grouped confidence intervals,
and CSI-error stress tests. The quantum-contribution target was **not demonstrated**.

Primary BER: MMSE 0.0604594; QAOA hybrid 0.0269656; uniform-sampling classical
hybrid 0.0268625. This is not evidence of quantum advantage. Local-simulator QAOA
p95 was 5.78 ms with four CPU workers; it is not a radio/QPU latency measurement.

See [report.json](confirmation-full/report.json),
[demo.html](confirmation-full/demo.html), and
[frozen.json](confirmation-full/frozen.json).
A post-test plot-legend cleanup is recorded in `presentation_update.json`; scores and
parameters were not recomputed or retuned. The bundled source snapshot was removed
during the 5G-only cleanup, so the saved confirmation report is available for review but
cannot be resumed with the exact historical source bundle.

`implementation-smoke` and `parallel-preflight` are plumbing checks, NOT confirmatory
evidence. Top-level older `signal_rescue.json` and its plots are preserved
historical/exploratory outputs. In particular, the old mMobile beam/subcarrier
axis remapping is NOT a physical multi-user MIMO channel and is excluded from v2 scores.
