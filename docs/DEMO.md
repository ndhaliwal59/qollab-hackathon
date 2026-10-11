# Signal Rescue live demo

## Start

From the repository root, run:

```sh
.venv/bin/python -m signal_rescue.local_app
```

The local server binds to `127.0.0.1:8000` and opens the app. Stop it with Ctrl-C.
If port 8000 is busy, pass `--port 8001`; use `--no-browser` to keep the current
browser unchanged. This runs the bundled Python code against Qiskit's `BasicSimulator`
and uses no cloud service.

## Judge flow

Start with eight streams and the classical preview. Press **Stress the Network** to
switch to twelve users sharing eight antennas. Adjust SNR, scenario, or seed and watch
the frame update; this preview does not execute quantum circuits. Press **Run Quantum
Decoder** to execute two adaptive rounds. The measured histogram, candidate objective,
common classical refinement, and six detectors all describe the same observation.
Click a detector row to highlight its user errors in the network.

The scene is a schematic uplink with one clear path per phone into the receiver
array. Its strength summarizes that user's actual channel power across all eight
receive antennas. Every moving dot follows its displayed line from user to receiver;
motion does not indicate continuous computation, propagation timing, or downlink
traffic. During the walkthrough, phone bits show the current recorded estimate,
and outlined phones/paths identify the correction block.

The decoder computes at full speed, then replays twelve recorded stages at a
readable pace: receive, MMSE estimate, classical starting point, and select/measure/
refine/decide for each of two quantum rounds, followed by the final estimate.
The panel labels this as a recorded walkthrough and reports actual computation time
separately. Backend failures preserve the incumbent and never invent measurements.
**Pause**, **Next step**, **Show final result**, and the replay speed selector control
only presentation. The header pause control also pauses the walkthrough and markers.

Choose **All seeds · 7–11** before running to compute each seed and finish its replay
before submitting the next seed. **Stop sweep** preserves completed runs and stops
further submissions; an in-flight backend call is allowed to finish. After the sweep,
click a seed chip to inspect that result. JSON export includes all completed seed
runs and their distinct execution receipts. Five diagnostic frames are separate from
the archived multi-frame benchmark and do not form a new benchmark result.

The seed and scenario intentionally reproduce the same observation. Local simulator
measurements also use a fixed seed, so repeated clicks can return identical counts
and detected bits while still executing new circuits. Each run displays a distinct
request identifier, completion time, executed circuit count, and measured shot count.
Change the seed or scenario to inspect a different frame. The scene explicitly
distinguishes a classical preview, a running decoder, and a completed run.

The first-round histogram compares equal candidate budgets for the same block and
incumbent. Round two is adaptive, so inspect its independent trace in the expandable
panel. A single frame is illustrative; the benchmark section contains the archived
aggregate evidence. The sampler uses a simulator, not quantum hardware.

Use **EXPORT RUN JSON** to save the recipe, source hashes, outputs, actual counts,
round traces, execution backend, and measured timings for the current run.

## Labels and caveats

The city and user paths are illustrative. User powers and error markers are linked to
the selected channel and detector. The channel slice is DeepMIMO ASU ray-traced data;
the transmitted bits and noise are synthesized for this experiment. Exact ML minimizes
the residual and is evaluated separately; it is not used to guide either hybrid.
QAOA improved over basic MMSE in the archived multi-frame study, while the matched
uniform classical hybrid performed similarly. These results do not demonstrate a
quantum advantage.
