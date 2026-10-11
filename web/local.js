import {createClient} from "./runtime/local.js";
import {mountApp} from "./ui/app.js";

await mountApp(createClient(), {runtimeLabel: "Live simulation", local: true});
