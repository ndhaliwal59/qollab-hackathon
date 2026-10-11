async function call(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : {"Content-Type": "application/json"},
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error(`Local server returned HTTP ${response.status} without JSON.`); }
  if (!response.ok) throw new Error(payload?.error || `Local server returned HTTP ${response.status}.`);
  return payload;
}

export function createClient() {
  return {
    bootstrap: () => call("/api/bootstrap"),
    preview: (spec) => call("/api/preview", spec),
    runQuantum: (spec) => call("/api/run", spec),
  };
}
