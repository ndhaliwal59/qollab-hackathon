# Qollab handoff

The project is structured so its Python experiment service and frontend can run in
Qollab without an application server. Generate the project files from the repository
root:

```sh
.venv/bin/python scripts/package_qollab.py
```

Upload the contents of `dist/qollab/` as a multi-file HTML/JavaScript project. Select a
built-in simulator backend. Its entry points are root-level `index.html`, `main.css`,
and `main.js`. The JavaScript transport imports `signal_rescue.live` and calls
`dispatch.callPromising(backend, action, payload_json)`; `runtime/qollab.js` keeps that
bridge behind the same client interface used by localhost.

The package includes literal Python data paths for the DeepMIMO slice and saved frozen
configuration/report. It omits NumPy observation archives, training and plotting code,
the local server, the virtual environment, and legacy datasets. The exporter checks the
100-file and 2 MB per-file limits and emits `PACKAGE_MANIFEST.json`.

## First platform smoke check

Before presenting the project, confirm in Qollab that:

1. All package files upload and Pyodide loads `signal_rescue.live` with its NumPy and
   Qiskit imports.
2. Bootstrap loads the channel slice, provenance, frozen configuration, and report.
3. The body markup, stylesheet, and SVG scene render at desktop and mobile widths.
4. A classical preview calls Python without submitting a circuit.
5. **Run Quantum Decoder** returns exactly 64 or 128 counts for round one and a second
   measured job for adaptive round two; the total must be twice the chosen per-round
   shots.
6. A second button-initiated run succeeds without restarting the project.
7. Stop/Run recovery and any Python package/runtime limits are checked in Qollab.

Qollab Python startup, packaged Qiskit/NumPy versions, file upload behavior, bridge
event handling, simulator latency, and public project behavior have not been verified
from this local environment. Do not substitute saved historical counts if live circuit
execution fails. The shared service reports a partial result with the classical
incumbent preserved when one or both circuit jobs fail.

## Backend note

The local browser calls a standard-library HTTP adapter and executes Qiskit on
`BasicSimulator`. Qollab's entry script instead awaits the supplied `backend` for each
adaptive round. The transport and frontend are shared; experiment construction,
sampling, and refinement remain in the Python service. Later hardware integrations
need a separately matched experiment and are outside this package.
