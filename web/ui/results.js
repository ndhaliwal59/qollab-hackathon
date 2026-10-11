const LABELS = {
  mmse: "First estimate", mmse_sic: "Subtract clear signals", classical: "Local search",
  uniform: "Random search", qaoa: "QAOA-based search", ml_reference: "Exact search",
};
const $ = (selector, root = document) => root.querySelector(selector);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = String(text);
  return n;
};
const SVG = "http://www.w3.org/2000/svg";

function renderCircuit(root, result = null, round = {}, samples = [], illustrative = false) {
  const host = $("#circuit-visual", root);
  const bits = round.bits || [];
  const count = bits.length || result?.model?.config?.k || 8;
  const fields = new Set(illustrative ? Array.from({length: count}, (_, i) => i) : (round.field_qubits || []));
  const pairs = illustrative
    ? Array.from({length: Math.floor(count / 2)}, (_, i) => [i * 2, i * 2 + 1])
    : (round.coupling_pairs || []);
  const candidateBits = samples?.[0]?.bits?.split("").reverse().join("") || "";
  $("#circuit-title", root).textContent = illustrative ? "Example circuit" : "Circuit from this run";
  $("#circuit-subtitle", root).textContent = illustrative
    ? "Choose a stage to highlight its gates."
    : "Select a stage to follow the gates used for this signal.";
  $("#circuit-mode", root).textContent = illustrative ? "Illustration" : "This run";
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
  title.textContent = illustrative ? "Example quantum circuit for signal decoding" : "Quantum circuit used for this signal decoding run";
  const desc = document.createElementNS(SVG, "desc"); desc.id = "circuit-svg-desc";
  desc.textContent = `${count} wires pass through Hadamard, mismatch score rotations, mixing rotations, and measurement. ${illustrative ? "This is a teaching example, not a circuit generated from the current settings." : `This run includes ${fields.size} single-bit rotations and ${pairs.length} pair interactions.`}`;
  svg.append(title, desc);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    svg.append(node); return node;
  };
  const xs = {prepare: 225, cost: 390, pair: 525, mix: 675, measure: 830};
  const y0 = 59, gap = 38;
  const groupNames = {prepare: "1 · Superposition", cost: "2 · Mismatch score", pair: `RZZ × ${pairs.length}`, mix: "3 · Mixer", measure: "4 · Readout"};
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
    add("text", {x: 892, y: y + 4, class: "circuit-sample", "text-anchor": "middle"}, candidateBits[i] || (illustrative ? "?" : "—"));
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
  add("text", {x: 892, y: y0 - 13, class: "circuit-sample-label", "text-anchor": "middle"}, "sample");
  if (pairs.length > shownPairs.length) add("text", {x: xs.pair, y: y0 + count * gap + 21, class: "circuit-column-label", "text-anchor": "middle"}, `+${pairs.length - shownPairs.length} pair gates bundled`);
  host.replaceChildren(svg);

  const explanations = {
    prepare: "H means Hadamard. It puts each selected bit into a quantum superposition, giving the circuit a starting spread of possible 0/1 combinations.",
    cost: illustrative
      ? "RZ represents how one phone bit changes the mismatch score. RZZ represents how a pair of bits changes it together. The example draws a few pairs so you can see the pattern."
      : `RZ adjusts individual bits; each RZZ gate links a pair whose combined choice changes the mismatch score. This run encoded ${fields.size} individual terms and ${pairs.length} pair interactions from the receiver's signal model.`,
    mix: "RX rotates every bit to explore different combinations. The circuit repeats this mismatch-score and mixing layer once in this demo.",
    measure: illustrative
      ? "M means measurement: each wire eventually gives a regular 0 or 1. The question marks stand for results you will see after running the decoder."
      : `Each M reads one wire as 0 or 1. The most frequent sample is ${candidateBits || "unavailable"}; the receiver then checks candidates on a regular computer.`,
  };
  const stageButtons = $("#circuit-stages", root);
  stageButtons.querySelectorAll("[data-circuit-stage]").forEach((item) => item.setAttribute("aria-pressed", "false"));
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
  $("#circuit-explanation", root).textContent = "The circuit prepares each bit, scores signal choices, explores alternatives, then measures guesses. Select a stage to highlight its gates.";
}

export function renderCircuitPreview(root) {
  renderCircuit(root, null, {}, [], true);
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
  rec.textContent = `${result.recipe.users} phones · signal clarity ${result.recipe.snr_db} dB · example ${result.recipe.scenario_index + 1}`;
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
    const displayBits = sample.bits.split("").reverse().join("");
    const bits = el("code", "histogram-bits", displayBits);
    const bar = el("span", "histogram-track");
    const fill = el("span", `histogram-fill ${kind}`); fill.style.width = `${Math.max(2, 100 * sample.count / max)}%`; bar.append(fill);
    const count = el("strong", "histogram-count", `${sample.count}×`);
    row.title = `Guess for highlighted phones in wire order: ${displayBits}; appeared ${sample.count} of ${total} times`;
    row.append(el("span", "rank", String(index + 1)), bits, bar, count);
    return row;
  }));
}

export function renderQuantum(root, result) {
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
  if (!rounds.length) trace.replaceChildren(el("p", "muted", "The QAOA-based search did not return a usable round; the best regular-computer guess remains available."));
}

export function renderBenchmark(root, bootstrap) {
  const benchmark = bootstrap.benchmark || {};
  // The page brief specifies this fixed 32,000-transmission presentation dataset.
  const rows = [
    {key: "mmse", name: "First estimate", value: "6.05%", ber: 6.05},
    {key: "mmse_sic", name: "Subtract clear signals", value: "3.30%", ber: 3.30},
    {key: "classical", name: "Local search", value: "3.21%", ber: 3.21},
    {key: "uniform", name: "Random search", value: "2.69%", ber: 2.69},
    {key: "qaoa", name: "QAOA-based search", value: "2.70%", ber: 2.70},
    {key: "ml_reference", name: "Exact search", value: "2.61%", ber: 2.61},
  ];
  const max = 6.05;
  for (const chart of root.querySelectorAll(".benchmark-bars")) chart.replaceChildren(...rows.map((row) => {
    const line = el("div", `benchmark-row${row.key === "qaoa" ? " benchmark-quantum" : ""}${row.key === "ml_reference" ? " benchmark-exact" : ""}`);
    const name = el("span", "benchmark-name", row.name);
    if (row.key === "ml_reference") name.append(el("small", "benchmark-best", "Best possible"));
    const track = el("span", "benchmark-track"), fill = el("span", `benchmark-fill ${row.key === "qaoa" ? "qaoa" : row.key === "uniform" ? "uniform" : "classical"}`);
    fill.style.width = `${100 * row.ber / max}%`; track.append(fill);
    line.append(name, track, el("strong", "benchmark-value", row.value));
    return line;
  }));
  const conclusion = $("#benchmark-conclusion", root);
  conclusion.textContent = "The reported averages differ by 0.01 percentage points. This test did not establish a quantum advantage.";
  const pairedRoot = $("#paired-evidence", root);
  pairedRoot.replaceChildren(
    el("span", "paired-label", "QAOA-based vs random search"),
    el("strong", "paired-difference", "0.01 percentage-point gap"),
  );
}

export function selectedMethod(root, detector) {
  return $("#detector-rows", root)?.querySelector(`[data-detector="${CSS.escape(detector)}"]`);
}

export {LABELS};
