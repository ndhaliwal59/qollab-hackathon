const bits = (symbols) => (symbols || []).map((value) => value < 0 ? "1" : "0").join("");

// This is a presentation of a completed trace. No detector or sampler runs here.
export function buildReplaySteps(result) {
  const method = (id) => result.methods.find((row) => row.id === id);
  const initial = method("mmse"), classical = method("classical"), final = method("qaoa");
  const steps = [
    {stage: "receive", title: "The signals arrive together", estimate: null, duration: 1400,
      description: `${result.recipe.users} users transmit ${result.truth_bits.length} bits through the same noisy channel at ${result.recipe.snr_db} dB. The receiver sees a mixture of all streams.`},
    {stage: "estimate", title: "Make a first guess", estimate: initial.bits, residual: initial.residual, duration: 1400,
      description: `The first estimate gets ${initial.bits_total - initial.errors} of ${initial.bits_total} bits right.`},
    {stage: "refine", title: "Check nearby answers", estimate: classical.bits, residual: classical.residual, duration: 1200,
      description: "The receiver tries small changes to its guess. Both candidate searches start from this same improved answer."},
  ];
  let estimate = classical.bits, residual = classical.residual, successful = 0;
  for (const [index, telemetry] of (result.qaoa_telemetry?.rounds || []).entries()) {
    const round = index + 1, selectedUsers = telemetry.bits || [];
    steps.push({stage: "select", round, selectedUsers, estimate, residual, duration: 1200,
      title: `Choose ${selectedUsers.length} hard-to-guess bits`,
      description: `The receiver revisits bits ${selectedUsers.map((user) => user + 1).join(", ")}. It holds the other bits steady while looking for a better combination.`});
    if (telemetry.fallback) {
      steps.push({stage: "decide", round, selectedUsers, estimate, residual, duration: 1600,
        title: `Round ${round}: preserve the starting estimate`,
        description: `The circuit did not return valid measurements. The decoder kept the previous estimate. ${telemetry.fallback}`});
      continue;
    }
    const trace = result.qaoa_replay?.rounds?.[successful];
    const samples = result.qaoa_rounds?.[successful] || [];
    successful++;
    const count = samples.reduce((sum, sample) => sum + sample.count, 0);
    steps.push({stage: "measure", round, selectedUsers, estimate, residual, samples, duration: 1800,
      title: `Sample possible fixes`,
      description: `The circuit produced ${count} guesses, with ${samples.length} different combinations. The most frequently sampled guesses appear below.`});
    steps.push({stage: "refine", round, selectedUsers, estimate, residual, duration: 1500,
      title: `Check the sampled fixes`,
      description: `A regular computer scores the sampled guesses and makes small changes to the strongest ${trace?.refined_candidates?.length || 0} candidates.`});
    if (trace?.incumbent_after) estimate = bits(trace.incumbent_after);
    else if (trace?.accepted && trace.refined_residuals?.length) {
      const winner = trace.refined_residuals.indexOf(Math.min(...trace.refined_residuals));
      estimate = bits(trace.refined_candidates[winner]);
    }
    residual = telemetry.residual_after;
    steps.push({stage: "decide", round, selectedUsers, estimate, residual, duration: 1400,
      title: telemetry.accepted ? `Round ${round}: accept the correction` : `Round ${round}: keep the previous estimate`,
      description: telemetry.accepted ? "The updated guess matches the received signal better, so the receiver keeps it." : "The new guesses did not improve the signal match, so the receiver keeps its previous answer."});
  }
  steps.push({stage: "complete", estimate: final.bits, residual: final.residual, duration: 1100,
    title: result.execution.status === "partial" ? "Partial run: final preserved estimate" : "Decoding complete",
    description: `${final.bits_total - final.errors} of ${final.bits_total} bits match the simulated answer. One frame is only an example, not a typical accuracy estimate.`});
  return steps;
}

// A controllable presentation clock. Pausing or skipping never alters computation.
export class ReplayPlayer {
  constructor({now = () => performance.now(), setTimer = (fn, delay) => setTimeout(fn, delay), clearTimer = (timer) => clearTimeout(timer)} = {}) {
    Object.assign(this, {now, setTimer, clearTimer, paused: false, playing: false, speed: 1});
  }
  play(steps, onStep, speed = 1) {
    this.cancel();
    Object.assign(this, {steps, onStep, index: 0, speed, playing: true});
    const completion = new Promise((resolve) => { this.resolve = resolve; });
    if (!steps.length) this.end(true);
    else this.show();
    return completion;
  }
  show() {
    this.remaining = this.steps[this.index].duration;
    this.onStep(this.steps[this.index], this.index, this.steps.length);
    this.schedule();
  }
  schedule() {
    if (!this.playing || this.paused) return;
    this.started = this.now();
    this.timer = this.setTimer(() => this.next(), this.speed ? this.remaining / this.speed : 0);
  }
  hold() {
    this.clearTimer(this.timer);
    if (!this.paused && this.playing) this.remaining = Math.max(0, this.remaining - (this.now() - this.started) * this.speed);
  }
  setPaused(paused) {
    if (this.paused === paused) return;
    this.hold(); this.paused = paused;
    if (!paused) this.schedule();
  }
  setSpeed(speed) {
    this.hold(); this.speed = speed; this.schedule();
  }
  next() {
    if (!this.playing) return;
    this.clearTimer(this.timer);
    if (++this.index >= this.steps.length) this.end(true);
    else this.show();
  }
  finish() {
    if (!this.playing) return;
    this.clearTimer(this.timer);
    this.index = this.steps.length - 1;
    this.onStep(this.steps[this.index], this.index, this.steps.length);
    this.end(true);
  }
  cancel() { if (this.playing) this.end(false); }
  end(completed) {
    this.clearTimer(this.timer); this.playing = false;
    this.resolve?.(completed); this.resolve = null;
  }
}

export async function runSeedSequence({seeds, makeRequest, compute, present, isStopped = () => false}) {
  const results = [];
  for (const seed of seeds) {
    if (isStopped()) break;
    const request = makeRequest(seed);
    const response = await compute(request);
    if (response.result?.request_id !== request.request_id) throw new Error("The backend returned a result for a different request.");
    results.push(response.result);
    await present(response.result, results.length - 1);
  }
  return results;
}

const STAGES = [["receive", "Receive"], ["estimate", "First guess"], ["select", "Pick bits"], ["measure", "Try guesses"], ["refine", "Check fixes"], ["decide", "Keep or undo"]];
const node = (tag, className, text) => {
  const element = document.createElement(tag); element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
};

export function renderReplay(root, result, step, index, total, previousEstimate = "") {
  const $ = (selector) => root.querySelector(selector);
  $("#playback-panel").hidden = false;
  $("#playback-receipt").textContent = "Step-by-step replay of the completed search";
  $("#playback-position").textContent = `${step.round ? `Round ${step.round} · ` : ""}Step ${index + 1} / ${total}`;
  $("#playback-title").textContent = step.title;
  $("#playback-description").textContent = step.description;
  $("#playback-residual").textContent = step.residual === undefined ? "" : `Signal fit ${Number(step.residual).toPrecision(4)}`;
  $("#decode-stages").replaceChildren(...STAGES.map(([id, label]) => {
    const item = node("li", id === step.stage || step.stage === "complete" ? "active" : "", label);
    if (id === step.stage) item.setAttribute("aria-current", "step");
    return item;
  }));
  const selected = new Set(step.selectedUsers || []);
  const rows = [["Transmitted", result.truth_bits], ["Current estimate", step.estimate]];
  $("#playback-bits").replaceChildren(...rows.map(([label, value]) => {
    const row = node("div", "replay-bit-row"); row.append(node("span", "replay-bit-label", label));
    const cells = node("div", "replay-bit-cells");
    [...result.truth_bits].forEach((truth, user) => {
      const bit = value?.[user];
      const cell = node("span", `replay-bit${selected.has(user) ? " in-block" : ""}${bit && bit !== truth ? " wrong" : ""}${label === "Current estimate" && bit && previousEstimate[user] && previousEstimate[user] !== bit ? " changed" : ""}`, bit || "·");
      cell.title = `User ${user + 1}${bit ? `: ${label.toLowerCase()} ${bit}` : ": not yet estimated"}`;
      cells.append(cell);
    });
    row.append(cells); return row;
  }));
  const candidates = $("#playback-candidates"); candidates.hidden = !step.samples?.length;
  candidates.replaceChildren(...(step.samples || []).slice(0, 6).map((sample) => {
    const item = node("div", "replay-sample");
    item.append(node("code", "", sample.bits), node("strong", "", `${sample.count}×`));
    const track = node("span", "replay-sample-track"), fill = node("span", "");
    fill.style.width = `${100 * sample.count / Math.max(...step.samples.map((entry) => entry.count))}%`;
    track.append(fill); item.append(track); return item;
  }));
}
