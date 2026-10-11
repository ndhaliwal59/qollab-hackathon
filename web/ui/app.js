import {renderNetwork, setNetworkMotion} from "./network.js";
import {renderBenchmark, renderCircuitPreview, renderComparison, renderQuantum} from "./results.js";
import {ReplayPlayer, buildReplaySteps, renderReplay, runSeedSequence} from "./replay.js";

const $ = (selector, root = document) => root.querySelector(selector);

export async function mountApp(client, {runtimeLabel = "Local Qiskit simulator", local = true} = {}) {
  const state = {users: 8, selected: "classical", preview: null, run: null, bootstrap: null, request: 0, latestPreviewId: null, activeRunId: null, activeRun: false, timer: null,
    player: new ReplayPlayer(), replayStep: null, runs: [], seeds: [], stopping: false, computing: false};
  const root = document;
  renderCircuitPreview(root);
  $("#runtime-label", root).textContent = runtimeLabel;
  const button = $("#run-button", root), runHint = $("#run-hint", root), banner = $("#status-banner", root);
  const controls = [$("#snr-control", root), $("#scenario-select", root), $("#seed-select", root),
    $("#shots-select", root), $("#next-scenario", root), $("#stress-button", root), $("#run-mode", root), ...root.querySelectorAll("[data-users]")].filter(Boolean);
  const setLocked = (locked) => { state.activeRun = locked; controls.forEach((node) => { node.disabled = locked; }); button.disabled = locked || !state.bootstrap; };
  const spec = () => ({request_id: `sr-${Date.now().toString(36)}-${++state.request}`, users: state.users,
    snr_db: Number($("#snr-control", root).value), scenario_index: Number($("#scenario-select", root).value),
    seed: Number($("#seed-select", root).value), shots_per_round: Number($("#shots-select", root).value)});
  const renderScene = (result = state.run || state.preview) => {
    const row = result?.methods?.find((item) => item.id === state.selected) || result?.methods?.find((item) => item.id === "classical");
    const step = state.replayStep;
    const detected = step ? step.estimate || "" : row?.bits || "";
    const errors = step ? [...detected].flatMap((bit, user) => bit !== result?.truth_bits?.[user] ? [user] : []) : row?.error_indices || [];
    renderNetwork($("#network-scene", root), {users: result?.recipe?.users || state.users,
      linkPowers: result?.recipe?.channel_link_powers || [], errors, sigma2: result?.recipe?.sigma2_complex ?? .1,
      truth: result?.truth_bits || "", detected,
      selectedUsers: step?.selectedUsers || [], phase: step?.stage || "idle",
      uncertainties: result?.recipe?.user_uncertainty || [],
      frame: result ? Number(result.recipe.scenario_index) + 1 : 0});
    const stage = state.computing ? "The receiver is searching for a better guess" : state.player.playing ? "Following the recorded steps" : state.run ? "Showing the decoded result" : result ? "Starting guesses" : "Loading example";
    $("#scene-status", root).textContent = stage;
    const overloaded = state.users > 8;
    $("#scene-title", root).textContent = overloaded ? "More phones than antennas." : "Phone signals overlap.";
    $("#scene-condition", root).textContent = `${state.users} phones / 8 antennas${overloaded ? " · crowded" : ""}`;
    const loadingHint = $("#loading-hint", root);
    if (loadingHint) loadingHint.textContent = overloaded ? "12 phones · 8 tower antennas" : "8 phones · 8 tower antennas";
  };
  const renderCurrent = () => {
    const result = state.run || state.preview;
    $("#comparison-title", root).textContent = "Compare the receiver’s guesses.";
    if (!result) {
      $("#bit-comparison", root).hidden = true;
      $("#preview-placeholder", root).hidden = false;
      $("#preview-placeholder p", root).textContent = "Calculating the receiver’s guesses…";
      return renderScene();
    }
    const activeRow = renderComparison(root, result, state.selected);
    renderScene(result);
    if (activeRow) $("#comparison-title", root).textContent = "Compare the receiver’s guesses.";
  };
  const invalidateRun = () => {
    state.run = null;
    state.preview = null;
    state.replayStep = null; state.runs = []; state.seeds = [];
    $("#playback-panel", root).hidden = true;
    state.latestPreviewId = null;
    $("#quantum-results", root).hidden = true;
    renderCircuitPreview(root);
    $("#comparison-title", root).textContent = "Compare the receiver’s guesses.";
    state.selected = "classical";
    renderCurrent();
  };
  let previewTimer;
  const preview = async () => {
    if (!state.bootstrap || state.activeRun) return;
    const request = spec();
    state.latestPreviewId = request.request_id;
    runHint.textContent = "Preparing the receiver’s guesses…";
    try {
      const response = await client.preview(request);
      if (response.result?.request_id !== request.request_id || state.latestPreviewId !== request.request_id || state.activeRun) return;
      state.preview = response.result;
      renderCurrent();
      runHint.textContent = "Starting guesses are ready.";
      banner.hidden = true;
    } catch (error) {
      if (state.latestPreviewId !== request.request_id || state.activeRun) return;
      runHint.textContent = "Preview could not be calculated.";
      banner.textContent = error.message || String(error); banner.hidden = false;
    }
  };
  const schedulePreview = () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(preview, 150);
  };
  try {
    state.bootstrap = await client.bootstrap();
    renderBenchmark(root, state.bootstrap);
    button.disabled = false;
    await preview();
  } catch (error) {
    runHint.textContent = "Application data could not be loaded.";
    banner.textContent = error.message || String(error); banner.hidden = false;
  }

  root.querySelectorAll("[data-users]").forEach((item) => item.addEventListener("click", () => {
    state.users = Number(item.dataset.users);
    root.querySelectorAll("[data-users]").forEach((button) => button.setAttribute("aria-pressed", String(button === item)));
    invalidateRun(); schedulePreview();
  }));
  $("#snr-control", root).addEventListener("input", (event) => {
    $("#snr-output", root).textContent = event.target.value;
    invalidateRun(); schedulePreview();
  });
  $("#scenario-select", root).addEventListener("change", () => {invalidateRun(); schedulePreview();});
  $("#seed-select", root).addEventListener("change", () => {invalidateRun(); schedulePreview();});
  $("#shots-select", root).addEventListener("change", () => {invalidateRun(); schedulePreview();});
  $("#next-scenario", root).addEventListener("click", () => {
    const select = $("#scenario-select", root); select.value = String((Number(select.value) + 1) % 8);
    invalidateRun(); schedulePreview();
  });
  $("#stress-button", root)?.addEventListener("click", () => {
    if (!state.bootstrap || state.activeRun) return;
    state.users = 12;
    root.querySelectorAll("[data-users]").forEach((item) => item.setAttribute("aria-pressed", String(Number(item.dataset.users) === 12)));
    invalidateRun();
    $("#run-button", root).click();
  });
  $("#detector-rows", root).addEventListener("click", (event) => {
    if (state.activeRun) return;
    const row = event.target.closest("[data-detector]");
    if (!row) return;
    state.replayStep = null;
    state.selected = row.dataset.detector; renderCurrent();
  });
  const playbackControls = () => {
    const count = state.player.steps?.length || 0;
    $("#replay-pause", root).disabled = !state.player.playing;
    $("#replay-prev", root).disabled = !count || state.player.index <= 0;
    $("#replay-next", root).disabled = !count || state.player.index >= count - 1;
    $("#replay-skip", root).disabled = !state.player.playing;
    $("#replay-pause", root).textContent = state.player.paused ? "Resume" : "Pause";
    $("#replay-pause", root).setAttribute("aria-pressed", String(state.player.paused));
  };
  const renderSeedProgress = () => {
    const progress = $("#seed-progress", root);
    progress.replaceChildren(...state.seeds.map((seed) => {
      const result = state.runs.find((run) => run.recipe.seed === seed);
      const active = Number($("#seed-select", root).value) === seed;
      const chip = document.createElement("button"); chip.type = "button";
      chip.className = "seed-chip"; chip.dataset.seed = seed;
      chip.dataset.state = result ? (active && state.player.playing ? "replaying" : "complete") : active && state.computing ? "computing" : "pending";
      chip.disabled = state.activeRun || !result;
      const final = result?.methods.find((row) => row.id === "qaoa");
      chip.textContent = `Seed ${seed} · ${active && state.player.playing ? "replaying" : result ? `${final.errors}/${final.bits_total} errors` : active && state.computing ? "computing" : "pending"}`;
      chip.setAttribute("aria-pressed", String(active));
      chip.addEventListener("click", () => {
        state.run = result; state.preview = result; state.selected = "qaoa";
        $("#seed-select", root).value = String(seed);
        const steps = buildReplaySteps(result);
        renderCurrent(); renderQuantum(root, result);
        state.player.select(steps, (step, position, total) => {
          state.replayStep = step;
          state.selected = step.stage === "estimate" ? "mmse" : step.stage === "complete" ? "qaoa" : step.round || step.stage === "receive" ? null : "classical";
          const previousEstimate = steps.slice(0, position).reverse().find((item) => item.estimate)?.estimate || "";
          renderReplay(root, result, step, position, total, previousEstimate);
          renderCurrent(); playbackControls(); renderSeedProgress();
        });
      });
      return chip;
    }));
  };
  const setPaused = (paused) => {
    document.body.classList.toggle("motion-paused", paused);
    state.player.setPaused(paused);
    const toggle = $("#motion-toggle", root);
    if (toggle) {
      toggle.setAttribute("aria-pressed", String(paused)); toggle.textContent = paused ? "Resume" : "Pause";
      toggle.setAttribute("aria-label", paused ? "Resume motion" : "Pause motion");
    }
    setNetworkMotion($("#network-scene", root), paused);
    playbackControls();
  };
  $("#replay-pause", root).addEventListener("click", () => setPaused(!state.player.paused));
  $("#replay-prev", root).addEventListener("click", () => state.player.previous());
  $("#replay-next", root).addEventListener("click", () => state.player.next());
  $("#replay-skip", root).addEventListener("click", () => state.player.finish());
  $("#replay-speed", root).addEventListener("change", (event) => state.player.setSpeed(Number(event.target.value)));
  $("#sweep-stop", root).addEventListener("click", () => {
    state.stopping = true; state.player.finish();
    $("#sweep-stop", root).disabled = true;
    runHint.textContent = state.computing ? "Finishing the current search…" : "Stopping the examples…";
  });
  button.addEventListener("click", async () => {
    if (!state.bootstrap || state.activeRun) return;
    state.player.reset();
    playbackControls();
    const base = spec();
    clearTimeout(previewTimer);
    state.latestPreviewId = null;
    state.run = null; state.replayStep = null; state.runs = []; state.stopping = false;
    state.seeds = $("#run-mode", root).value === "all" ? state.bootstrap.supported.seeds : [base.seed];
    state.selected = "classical";
    setLocked(true); banner.hidden = true;
    $("#sweep-stop", root).hidden = state.seeds.length === 1;
    $("#sweep-stop", root).disabled = false;
    $("#quantum-results", root).hidden = true;
    renderCircuitPreview(root);
    $(".lab-card", root).scrollIntoView({behavior: "smooth", block: "start"});
    try {
      await runSeedSequence({
        seeds: state.seeds, isStopped: () => state.stopping,
        makeRequest: (seed) => ({...base, seed, request_id: spec().request_id}),
        compute: async (request) => {
          state.activeRunId = request.request_id; state.computing = true;
          state.replayStep = null;
          if (state.preview?.recipe.seed !== request.seed) state.preview = null;
          state.run = null; $("#seed-select", root).value = String(request.seed);
          renderCurrent(); playbackControls(); renderSeedProgress();
          $("#playback-panel", root).hidden = false;
          $("#playback-receipt", root).textContent = `Computing live circuits · seed ${request.seed}`;
          $("#playback-title", root).textContent = "Finding candidate fixes";
          $("#playback-description", root).textContent = "The receiver is searching for better guesses. The step-by-step replay starts when it is done.";
          $("#playback-position", root).textContent = ""; $("#playback-residual", root).textContent = "";
          for (const id of ["decode-stages", "playback-bits", "playback-candidates"]) $(`#${id}`, root).replaceChildren();
          $("#quantum-results", root).hidden = true;
          renderCircuitPreview(root);
          const started = performance.now();
          state.timer = setInterval(() => {runHint.textContent = `Searching for candidate fixes · ${((performance.now() - started) / 1000).toFixed(1)} s`;}, 500);
          try { return await client.runQuantum(request); }
          finally { clearInterval(state.timer); state.timer = null; state.computing = false; }
        },
        present: async (result) => {
          state.runs.push(result); state.run = result; state.preview = result; state.selected = "qaoa";
          renderQuantum(root, result);
          if (result.execution.warning) {banner.textContent = result.execution.warning; banner.hidden = false;}
          const steps = buildReplaySteps(result);
          const showStep = (step, position, total) => {
            state.replayStep = step;
            state.selected = step.stage === "estimate" ? "mmse" : step.stage === "complete" ? "qaoa" : step.round || step.stage === "receive" ? null : "classical";
            const previousEstimate = steps.slice(0, position).reverse().find((item) => item.estimate)?.estimate || "";
            renderReplay(root, result, step, position, total, previousEstimate);
            renderCurrent(); playbackControls(); renderSeedProgress();
            runHint.textContent = `Following step ${position + 1} of ${total}`;
          };
          if (state.stopping) showStep(steps.at(-1), steps.length - 1, steps.length);
          else await state.player.play(steps, showStep, Number($("#replay-speed", root).value));
        },
      });
      runHint.textContent = state.stopping ? "Stopped. Completed examples are still available." : "Walkthrough complete.";
    } catch (error) {
      state.player.cancel();
      runHint.textContent = "The search stopped. Try again when ready.";
      banner.textContent = error.message || String(error); banner.hidden = false;
    } finally {
      clearInterval(state.timer); state.timer = null; setLocked(false);
      state.activeRunId = null; state.computing = false;
      if (!state.run && state.runs.length) {
        state.run = state.runs.at(-1); state.preview = state.run; state.selected = "qaoa";
        $("#seed-select", root).value = String(state.run.recipe.seed); renderQuantum(root, state.run);
        const steps = buildReplaySteps(state.run); state.replayStep = steps.at(-1);
        renderReplay(root, state.run, state.replayStep, steps.length - 1, steps.length);
      } else if (!state.run) {
        $("#playback-receipt", root).textContent = "No completed search";
        $("#playback-title", root).textContent = "The decoder could not complete this request";
        $("#playback-description", root).textContent = "The receiver could not finish this search. Choose Run decoder to try again.";
      }
      renderCurrent(); playbackControls(); renderSeedProgress();
      $("#sweep-stop", root).hidden = true;
    }
  });
  $("#motion-toggle", root)?.addEventListener("click", () => setPaused(!state.player.paused));
  return state;
}
