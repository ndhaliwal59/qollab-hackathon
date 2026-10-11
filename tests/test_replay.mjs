import test from "node:test";
import assert from "node:assert/strict";
import {ReplayPlayer, buildReplaySteps, runSeedSequence} from "../web/ui/replay.js";

function fakeClock() {
  let time = 0, id = 0;
  const tasks = new Map();
  return {
    now: () => time,
    setTimer: (fn, delay) => {tasks.set(++id, {fn, at: time + delay}); return id;},
    clearTimer: (key) => tasks.delete(key),
    advance: (ms) => {
      const end = time + ms;
      for (let count = 0; count < 1000; count++) {
        const next = [...tasks.entries()].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        tasks.delete(next[0]); time = next[1].at; next[1].fn();
      }
      time = end;
    },
  };
}
const steps = [{title: "first", duration: 1000}, {title: "last", duration: 1000}];

test("pause freezes the stage clock and manual stepping stays paused", async () => {
  const clock = fakeClock(), player = new ReplayPlayer(clock), shown = [];
  const finished = player.play(steps, (step) => shown.push(step.title));
  clock.advance(300); player.setPaused(true); clock.advance(5000);
  assert.deepEqual(shown, ["first"]);
  player.next(); clock.advance(5000);
  assert.deepEqual(shown, ["first", "last"]);
  assert.equal(player.playing, true);
  player.setPaused(false); clock.advance(1000);
  assert.equal(await finished, true);
  assert.equal(player.playing, false);
});

test("speed changes preserve the unplayed portion of a stage", async () => {
  const clock = fakeClock(), player = new ReplayPlayer(clock), shown = [];
  const finished = player.play(steps, (step) => shown.push(step.title));
  clock.advance(300); player.setSpeed(2); clock.advance(349);
  assert.deepEqual(shown, ["first"]);
  clock.advance(1); assert.deepEqual(shown, ["first", "last"]);
  player.finish(); assert.equal(await finished, true);
});

test("skip shows the actual final step and clears pending timers", async () => {
  const clock = fakeClock(), player = new ReplayPlayer(clock), shown = [];
  const finished = player.play(steps, (step) => shown.push(step.title));
  player.finish(); clock.advance(10000);
  assert.deepEqual(shown, ["first", "last"]);
  assert.equal(await finished, true);
});

test("cancelling resolves the old playback without advancing it", async () => {
  const clock = fakeClock(), player = new ReplayPlayer(clock), shown = [];
  const first = player.play(steps, (step) => shown.push(step.title));
  const second = player.play(steps, (step) => shown.push(step.title), 0);
  assert.equal(await first, false);
  clock.advance(0); assert.equal(await second, true);
  assert.deepEqual(shown, ["first", "first", "last"]);
});

function result() {
  return {
    recipe: {users: 2, snr_db: 12}, truth_bits: "01",
    execution: {status: "complete", circuit_invocations: 2, shots: 4},
    methods: [
      {id: "mmse", bits: "00", residual: 5, errors: 1, bits_total: 2},
      {id: "classical", bits: "00", residual: 4, errors: 1, bits_total: 2},
      {id: "qaoa", bits: "01", residual: 2, errors: 0, bits_total: 2},
    ],
    qaoa_telemetry: {rounds: [
      {bits: [1], accepted: true, residual_before: 4, residual_after: 2},
      {bits: [0], accepted: false, residual_before: 2, residual_after: 2},
    ]},
    qaoa_rounds: [[{bits: "1", count: 2}], [{bits: "0", count: 2}]],
    qaoa_replay: {rounds: [
      {incumbent_after: [1, -1], refined_candidates: [[1, -1]], refined_residuals: [2]},
      {incumbent_after: [1, -1], refined_candidates: [[-1, -1]], refined_residuals: [3]},
    ]},
  };
}

test("walkthrough uses measured counts and authoritative accepted incumbents", () => {
  const data = result(), trace = buildReplaySteps(data);
  assert.deepEqual(trace.filter((step) => step.stage === "decide").map((step) => step.estimate), ["01", "01"]);
  assert.equal(trace.find((step) => step.stage === "measure").samples, data.qaoa_rounds[0]);
  assert.deepEqual(trace.find((step) => step.stage === "select").selectedUsers, [1]);
  assert.equal(trace.at(-1).estimate, data.methods.at(-1).bits);
  assert.equal(trace.at(-1).residual, 2);
});

test("failed rounds preserve the estimate and do not invent measurements", () => {
  const data = result();
  data.qaoa_telemetry.rounds = [{bits: [1], fallback: "backend unavailable", residual_after: 4}];
  data.qaoa_rounds = []; data.qaoa_replay.rounds = [];
  data.methods.at(-1).bits = "00";
  const trace = buildReplaySteps(data);
  assert.equal(trace.filter((step) => step.stage === "measure").length, 0);
  assert.equal(trace.find((step) => step.stage === "decide").estimate, "00");
  assert.match(trace.find((step) => step.stage === "decide").description, /backend unavailable/);
});

test("all seeds compute sequentially, each after the preceding replay finishes", async () => {
  const events = [];
  const results = await runSeedSequence({
    seeds: [7, 8, 9, 10, 11], makeRequest: (seed) => ({seed, request_id: `seed-${seed}`}),
    compute: async (request) => {events.push(`compute ${request.seed}`); return {result: request};},
    present: async (run) => {await Promise.resolve(); events.push(`replay ${run.seed}`);},
  });
  assert.deepEqual(events, [7, 8, 9, 10, 11].flatMap((seed) => [`compute ${seed}`, `replay ${seed}`]));
  assert.equal(new Set(results.map((run) => run.request_id)).size, 5);
});

test("stop keeps the completed seed and submits no following seeds", async () => {
  let stopped = false;
  const results = await runSeedSequence({
    seeds: [7, 8, 9], makeRequest: (seed) => ({seed, request_id: String(seed)}),
    compute: async (request) => ({result: request}),
    present: async () => {stopped = true;}, isStopped: () => stopped,
  });
  assert.deepEqual(results.map((run) => run.seed), [7]);
});

test("mismatched backend responses never enter the visual replay", async () => {
  await assert.rejects(runSeedSequence({
    seeds: [7], makeRequest: () => ({request_id: "new-run"}),
    compute: async () => ({result: {request_id: "stale-run"}}),
    present: () => assert.fail("Stale data must not be replayed"),
  }), /different request/);
});
