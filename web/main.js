// Qollab entry. The runtime supplies `backend` before main.js runs.
import {createClient} from "./runtime/qollab.js";
import {mountApp} from "./ui/app.js";

await mountApp(createClient(backend), {runtimeLabel: "Live simulation", local: false});
// Keep this Qollab run alive so event handlers can submit a second adaptive
// correction circuit and the interactive page can continue receiving results.
await new Promise(() => {});
