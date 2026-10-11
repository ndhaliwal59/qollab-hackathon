const LABELS = {
  mmse: "First estimate", mmse_sic: "Subtract clear signals", classical: "Local search",
  uniform: "Random search", qaoa: "Quantum-assisted search", ml_reference: "Exact search (reference)",
};
const $ = (selector, root = document) => root.querySelector(selector);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = String(text);
  return n;
};
const SVG = "http://www.w3.org/2000/svg";

function renderCircuit(root, result, round, samples) {
  const host = $("#circuit-visual", root);
  const bits = round.bits || [];
  const count = bits.length || result.model?.config?.k || 8;
  const fields = new Set(round.field_qubits || []);
  const pairs = round.coupling_pairs || [];
  const candidate = samples?.[0]?.bits || "—";
  $("#circuit-qubit-count", root).textContent = `${count} uncertain bits`;
  const shownPairs = [];
  const occupied = new Set();
  for (const pair of pairs) {
    const [a, b] = pair;
    if (shownPairs.length >= 4) break;
    if (a < count && b < count && !occupied.has(a) && !occupied.has(b)) {
      shownPairs.push(pair); occupied.add(a); occupied.add(b);
    }
  }
  const height = Math.max(210, 98 + count * 38);
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("viewBox", `0 0 930 ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-labelledby", "circuit-svg-title circuit-svg-desc");
  const title = document.createElementNS(SVG, "title"); title.id = "circuit-svg-title";
  title.textContent = "Quantum circuit used for this signal decoding run";
  const desc = document.createElementNS(SVG, "desc"); desc.id = "circuit-svg-desc";
  desc.textContent = `${count} wires pass through Hadamard, signal cost rotations, mixing rotations, and measurement. The circuit includes ${fields.size} single-bit rotations and ${pairs.length} pair interactions.`;
  svg.append(title, desc);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node); return node;
  };
  const xs = {prepare: 225, cost: 390, pair: 525, mix: 675, measure: 830};
  const y0 = 59, gap = 38;
  const groupNames = {prepare: "1 · Superposition", cost: "2 · Signal cost", pair: `RZZ × ${pairs.length}`, mix: "3 · Mixer", measure: "4 · Readout"};
  for (const [key, label] of Object.entries(groupNames)) add("text", {x: xs[key], y: 20, class: "circuit-column-label", "text-anchor": "middle"}, label);
  for (let i = 0; i < count; i++) {
    const y = y0 + i * gap;
    const phone = bits.length ? bits[i] + 1 : i + 1;
    add("text", {x: 18, y: y + 4, class: "circuit-wire-label"}, `bit ${i + 1} · phone ${phone}`);
    add("line", {x1: 162, y1: y, x2: 858, y2: y, class: "circuit-wire"});
    add("rect", {x: xs.prepare - 17, y: y - 15, width: 34, height: 30, rx: 2, class: "circuit-gate stage-prepare"});
    add("text", {x: xs.prepare, y: y + 4, class: "circuit-gate-text stage-prepare", "text-anchor": "middle"}, "H");
    if (fields.has(i)) {
      add("rect", {x: xs.cost - 17, y: y - 15, width: 34, height: 30, rx: 2, class: "circuit-gate stage-cost"});
      add("text", {x: xs.cost, y: y + 4, class: "circuit-gate-text stage-cost", "text-anchor": "middle"}, "RZ");
    }
    add("rect", {x: xs.mix - 17, y: y - 15, width: 34, height: 30, rx: 2, class: "circuit-gate stage-mix"});
    add("text", {x: xs.mix, y: y + 4, class: "circuit-gate-text stage-mix", "text-anchor": "middle"}, "RX");
    add("rect", {x: xs.measure - 13, y: y - 13, width: 26, height: 26, rx: 13, class: "circuit-meter stage-measure"});
    add("text", {x: xs.measure, y: y + 4, class: "circuit-meter-text stage-measure", "text-anchor": "middle"}, "M");
  }
  for (const pair of shownPairs) {
    const [a, b] = pair;
    const ya = y0 + a * gap, yb = y0 + b * gap, middle = (ya + yb) / 2;
    add("path", {d: `M ${xs.pair} ${ya} L ${xs.pair} ${yb}`, class: "circuit-coupling stage-cost"});
    add("circle", {cx: xs.pair, cy: ya, r: 3, class: "circuit-pair-dot stage-cost"});
    add("circle", {cx: xs.pair, cy: yb, r: 3, class: "circuit-pair-dot stage-cost"});
    add("rect", {x: xs.pair - 21, y: middle - 8, width: 42, height: 16, rx: 2, class: "circuit-pair-gate stage-cost"});
    add("text", {x: xs.pair, y: middle + 3, class: "circuit-pair-text stage-cost", "text-anchor": "middle"}, "RZZ");
  }
  add("text", {x: xs.measure, y: y0 - 13, class: "circuit-sample-label", "text-anchor": "middle"}, "sample");
  add("text", {x: xs.measure, y: y0 + 4, class: "circuit-sample", "text-anchor": "middle"}, candidate);
  if (pairs.length > shownPairs.length) add("text", {x: xs.pair, y: y0 + count * gap + 21, class: "circuit-column-label", "text-anchor": "middle"}, `+${pairs.length - shownPairs.length} pair gates bundled`);
  host.replaceChildren(svg);

  const explanations = {
    prepare: "H means Hadamard. It puts each selected bit into a quantum superposition, giving the circuit a starting spread of possible 0/1 combinations.",
    cost: `RZ adjusts individual bits; each RZZ gate links a pair whose combined choice affects signal fit. This run encoded ${fields.size} individual terms and ${pairs.length} pair interactions from the receiver's signal model.`,
    mix: "RX rotates every bit to explore different combinations. The circuit repeats this cost-and-mix layer once in this demo.",
    measure: `Measurement turns the circuit state into ordinary 0/1 candidates. The most frequent sample here is ${candidate}; the receiver then checks candidates on a regular computer.`,
  };
  const stageButtons = $("#circuit-stages", root);
  stageButtons.onclick = (event) => {
    const button = event.target.closest("[data-circuit-stage]");
    if (!button) return;
    const selected = button.dataset.circuitStage;
    stageButtons.querySelectorAll("[data-circuit-stage]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    svg.querySelectorAll(".stage-prepare, .stage-cost, .stage-mix, .stage-measure").forEach((node) => {
      node.classList.toggle("circuit-dim", !node.classList.contains(`stage-${selected}`));
    });
    $("#circuit-explanation", root).textContent = explanations[selected];
  };
  stageButtons.querySelector('[data-circuit-stage="prepare"]')?.click();
}

export function renderComparison(root, result, selected = "qaoa") {
  const methods = result.methods || [];
  const truth = result.truth_bits || "";
  const truthRow = $("#truth-row", root), rows = $("#detector-rows", root);
  truthRow.replaceChildren(...[...truth].map((bit, index) => {
    const item = el("span", "bit", bit); item.title = `User ${index + 1}: transmitted ${bit}`; return item;
  }));
  rows.replaceChildren(...methods.map((method) => {
    const line = el("button", `detector-row${method.id === selected ? " selected" : ""}`);
    line.type = "button"; line.dataset.detector = method.id;
    line.setAttribute("aria-pressed", String(method.id === selected));
    line.setAttribute("aria-label", `${LABELS[method.id]}: ${method.errors} errors in ${method.bits_total} bits`);
    const name = el("span", "detector-name", LABELS[method.id]);
    if (method.id === "ml_reference") name.title = "Exact maximum-likelihood minimizes the detection residual over all BPSK vectors; it does not necessarily minimize true-bit errors for this frame.";
    const bits = el("span", "bit-cells");
    [...method.bits].forEach((bit, index) => {
      const cell = el("span", `bit${bit !== truth[index] ? " error" : ""}`, bit);
      cell.title = `User ${index + 1}: detected ${bit}, transmitted ${truth[index]}`; bits.append(cell);
    });
    const frame = el("span", `frame-stat${method.bler ? " has-error" : ""}`);
    frame.append(el("strong", "", `${method.errors} / ${method.bits_total}`), el("span", "frame-track", ""));
    const fill = el("span", "frame-fill"); fill.style.width = `${Math.max(3, 100 * method.frame_ber)}%`;
    $(".frame-track", frame).append(fill);
    const residual = el("span", "residual-value", Number(method.residual).toPrecision(4));
    line.append(name, bits, frame, residual);
    return line;
  }));
  const rec = $("#result-recipe", root);
  rec.textContent = `${result.recipe.users} phones · clarity ${result.recipe.snr_db} dB · example ${result.recipe.scenario_index + 1}`;
  rec.title = result.recipe.group;
  $("#preview-placeholder", root).hidden = true;
  $("#bit-comparison", root).hidden = false;
  return methods.find((item) => item.id === selected) || methods[0];
}

export function renderHistogram(container, samples, total, kind) {
  if (!Array.isArray(samples) || !samples.length) {
    container.replaceChildren(el("p", "muted", "No measured candidates were returned for this round.")); return;
  }
  const max = Math.max(...samples.map((row) => row.count), 1);
  container.replaceChildren(...samples.slice(0, 8).map((sample, index) => {
    const row = el("div", "histogram-row");
    const bits = el("code", "histogram-bits", sample.bits);
    const bar = el("span", "histogram-track");
    const fill = el("span", `histogram-fill ${kind}`); fill.style.width = `${Math.max(2, 100 * sample.count / max)}%`; bar.append(fill);
    const count = el("strong", "histogram-count", `${sample.count}×`);
    row.title = `Candidate ${sample.bits}; sampled ${sample.count} of ${total} times; score ${Number(sample.energy).toFixed(4)}`;
    row.append(el("span", "rank", String(index + 1)), bits, bar, count);
    return row;
  }));
}

export function renderQuantum(root, result) {
  $("#quantum-placeholder", root).hidden = true;
  $("#quantum-results", root).hidden = false;
  const execution = result.execution || {};
  const card = $("#execution-card", root);
  card.className = `execution-card ${execution.status === "partial" ? "partial" : "complete"}`;
  card.replaceChildren();
  const title = el("strong", "execution-title", execution.status === "partial"
    ? "Some samples were unavailable; the best valid guess is kept."
    : "Candidate fixes sampled. Compare them with the random search.");
  card.append(title);
  if (execution.warning) card.append(el("p", "execution-warning", execution.warning));
  const gateData = result.qaoa_telemetry?.rounds?.[0] || {};
  const gates = Object.entries(gateData.gate_summary || {}).map(([name, count]) => `${name} × ${count}`).join(" · ");
  const details = $("#execution-details-content", root);
  const detailsLines = [
    `${execution.backend || "Quantum simulator"} · ${execution.circuit_invocations || 0} circuit runs · ${execution.shots || 0} samples`,
    `Signal example ${result.recipe.scenario_index + 1} · repeat seed ${result.recipe.seed}`,
    `${result.model?.config?.k || 8} qubits · circuit depth ${gateData.circuit_depth || 0}${gates ? ` · gates ${gates}` : ""}`,
    "This is a local classical simulation of a quantum circuit, not a quantum computer or a live mobile network.",
  ];
  details.replaceChildren(...detailsLines.map((line) => el("p", "technical-line", line)));
  const firstUniform = result.uniform_rounds?.[0] || [];
  const firstQaoa = result.qaoa_rounds?.[0] || [];
  const shots = result.model?.config?.shots || result.comparison?.shots_per_method_per_round || 64;
  $("#uniform-shot-label", root).textContent = `${shots} candidates`;
  $("#qaoa-shot-label", root).textContent = `${shots} shots`;
  renderHistogram($("#uniform-histogram", root), firstUniform, shots, "uniform");
  renderHistogram($("#qaoa-histogram", root), firstQaoa, shots, "qaoa");
  const trace = $("#trace-content", root);
  const rounds = result.qaoa_replay?.rounds || [];
  renderCircuit(root, result, result.qaoa_telemetry?.rounds?.[0] || {}, firstQaoa);
  trace.replaceChildren(...rounds.map((round, i) => {
    const section = el("article", "trace-round");
    section.append(el("h4", "", `Pass ${i + 1} · phones ${round.bits.map((bit) => bit + 1).join(", ")}`));
    const candidates = (round.sampled_candidates || []).length;
    section.append(el("p", "", `The circuit sampled ${shots} possible fixes. The receiver checked ${candidates} of them more closely.`));
    section.append(el("p", `trace-accept${round.accepted ? " yes" : ""}`, round.accepted
      ? "A fix matched the signal better, so the receiver kept it."
      : "None of the tested fixes matched better, so the receiver kept its previous guess."));
    return section;
  }));
  if (!rounds.length) trace.replaceChildren(el("p", "muted", "No successful QAOA round was returned; the valid classical incumbent remains available."));
}

export function renderBenchmark(root, bootstrap) {
  const benchmark = bootstrap.benchmark || {};
  const summary = benchmark.summary || {};
  const preferred = ["mmse", "mmse_sic", "classical", "uniform", "qaoa", "ml_reference"];
  const names = {mmse: "First estimate", mmse_sic: "Subtract clear signals", classical: "Local search", uniform: "Random search", qaoa: "Quantum search", ml_reference: "Exact search"};
  const rows = preferred.map((key) => {
    const item = summary[key] || {};
    return {key, name: names[key], ber: Number(item.ber ?? item.BER ?? item.bit_error_rate ?? item.primary_ber)};
  }).filter((item) => Number.isFinite(item.ber));
  const chart = $("#benchmark-bars", root), max = Math.max(...rows.map((row) => row.ber), 1e-9);
  chart.replaceChildren(...rows.map((row) => {
    const line = el("div", `benchmark-row${row.key === "qaoa" ? " benchmark-quantum" : ""}`);
    line.append(el("span", "benchmark-name", row.name));
    const track = el("span", "benchmark-track"), fill = el("span", `benchmark-fill ${row.key === "qaoa" ? "qaoa" : row.key === "uniform" ? "uniform" : "classical"}`);
    fill.style.width = `${100 * row.ber / max}%`; track.append(fill);
    line.append(track, el("strong", "benchmark-value", `${(row.ber * 100).toFixed(2)}%`));
    return line;
  }));
  const conclusion = $("#benchmark-conclusion", root);
  conclusion.textContent = "The quantum and matched random searches performed about the same. This experiment did not show a quantum advantage.";
  const paired = benchmark.comparisons || {};
  const qaoaUniform = paired.uniform || null;
  const pairedRoot = $("#paired-evidence", root);
  const ci = qaoaUniform?.ber_improvement_ci95 || [-0.0003369945, 0.0001284943];
  const includesNoDifference = ci[0] <= 0 && ci[1] >= 0;
  pairedRoot.replaceChildren(
    el("span", "paired-label", "Quantum vs random search"),
    el("strong", "paired-difference", "No clear winner"),
    el("span", "paired-interval", includesNoDifference ? "The uncertainty range includes no difference." : "The measured difference is outside the uncertainty range."),
  );
}

export function selectedMethod(root, detector) {
  return $("#detector-rows", root)?.querySelector(`[data-detector="${CSS.escape(detector)}"]`);
}

export {LABELS};
