import * as service from "signal_rescue.live";

function unpack(raw) {
  let value;
  try { value = JSON.parse(String(raw)); }
  catch { throw new Error("Qollab returned an unreadable experiment response."); }
  if (value.status !== "ok" && value.status !== "ready") {
    throw new Error(value.error || "The experiment could not be completed.");
  }
  return value;
}

export function createClient(backend) {
  return {
    bootstrap: async () => unpack(await service.dispatch.callPromising(backend, "bootstrap", "{}")),
    preview: async (spec) => unpack(await service.dispatch.callPromising(backend, "preview", JSON.stringify(spec))),
    runQuantum: async (spec) => unpack(await service.dispatch.callPromising(backend, "run", JSON.stringify(spec))),
  };
}
