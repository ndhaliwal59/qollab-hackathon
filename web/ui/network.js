const NS = "http://www.w3.org/2000/svg";
const ANTENNAS = [
  [618, 211], [630, 211], [642, 211], [654, 211],
  [666, 211], [678, 211], [690, 211], [702, 211],
];

function node(name, attributes = {}, text) {
  const el = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attributes)) el.setAttribute(key, String(value));
  if (text !== undefined) el.textContent = text;
  return el;
}

export function setNetworkMotion(container, paused) {
  const svg = container.querySelector("svg");
  if (!svg) return;
  if (paused) svg.pauseAnimations();
  else svg.unpauseAnimations();
}

export function renderNetwork(container, {users = 8, linkPowers = [], errors = [], uncertainties = [], truth = "", detected = "", sigma2 = 0.1, frame = 0, selectedUsers = [], phase = "idle"} = {}) {
  const reducedMotion = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches || false;
  const svg = node("svg", {viewBox: "0 0 1000 410", role: "presentation", "aria-hidden": "true", preserveAspectRatio: "xMidYMid meet"});
  const defs = node("defs");
  const glow = node("filter", {id: "signal-glow", x: "-120%", y: "-120%", width: "340%", height: "340%"});
  glow.append(node("feGaussianBlur", {stdDeviation: "3", result: "blur"}));
  const merge = node("feMerge"); merge.append(node("feMergeNode", {in: "blur"}), node("feMergeNode", {in: "SourceGraphic"})); glow.append(merge); defs.append(glow);
  svg.append(defs, node("rect", {x: 0, y: 0, width: 1000, height: 410, fill: "#fbfcfa"}));
  const grid = node("g", {stroke: "#e5ebea", "stroke-width": 1});
  for (let x = 0; x <= 1000; x += 50) grid.append(node("path", {d: `M${x} 0 V410`}));
  for (let y = 0; y <= 400; y += 50) grid.append(node("path", {d: `M0 ${y} H1000`}));
  svg.append(grid, node("path", {d: "M0 353 H1000", stroke: "#c6d1d4", "stroke-width": 1}));

  // Draw one clear illustrative link per phone into the receiver array.
  const usersPositions = Array.from({length: 12}, (_, i) => {
    const angle = (-156 + i * 25) * Math.PI / 180;
    return [498 + 414 * Math.cos(angle), 209 + 136 * Math.sin(angle)];
  });
  const selectedErrors = new Set(Array.isArray(errors) ? errors.map(Number) : []);
  const block = new Set(selectedUsers);
  const powerAt = (antenna, user) => {
    const value = Number(linkPowers[antenna]?.[user]);
    return Number.isFinite(value) && value > 0 ? value : 0;
  };
  const userPowers = Array.from({length: users}, (_, user) =>
    ANTENNAS.reduce((sum, _, antenna) => sum + powerAt(antenna, user), 0) / ANTENNAS.length);
  const maximum = Math.max(...userPowers, 1e-12);
  const links = node("g", {fill: "none", "stroke-linecap": "round"});
  const packets = node("g", {class: "signal-packets"});
  for (let user = 0; user < users; user++) {
    const [ux, uy] = usersPositions[user];
    {
      const power = userPowers[user];
      if (!power) continue;
      const [ax, ay] = [660, 211];
      const strength = Math.sqrt(power / maximum);
      const color = "#087c88";
      const id = `signal-u${user}`;
      const bend = 24 + (user % 3) * 12;
      const path = node("path", {id, d: `M${ux.toFixed(1)} ${(uy - 12).toFixed(1)} Q${((ux + ax) / 2).toFixed(1)} ${(Math.min(uy - 12, ay) - bend).toFixed(1)} ${ax} ${ay + 15}`,
        stroke: color, "stroke-width": .6 + 1.4 * strength, opacity: .08 + .38 * strength,
        class: `signal-path${block.has(user) ? " block-selected" : block.size ? " block-muted" : ""}`, "data-user": user, "data-power": power});
      path.append(node("title", {}, `U${user + 1} → receiver antenna array: average |H|² = ${power.toPrecision(4)}`));
      links.append(path);
      if (!reducedMotion && (phase === "idle" || phase === "receive")) {
        const packet = node("circle", {r: .9 + .9 * strength, fill: color, opacity: .12 + .7 * strength,
          class: "signal-packet", "data-user": user});
        // Animation timing is illustrative; it does not encode propagation delay.
        const motion = node("animateMotion", {dur: phase === "receive" ? "1.2s" : "3s", begin: phase === "receive" ? "0s" : `${-(user * .23).toFixed(2)}s`, repeatCount: phase === "receive" ? "1" : "indefinite", fill: "freeze", calcMode: "linear"});
        motion.append(node("mpath", {href: `#${id}`}));
        packet.append(motion); packets.append(packet);
      }
    }
  }
  svg.append(links, packets);

  // Antenna mast, subtle signal rings, equipment cabinet and indicator lights.
  const tower = node("g", {transform: "translate(0,0)"});
  tower.append(node("ellipse", {cx: 660, cy: 365, rx: 115, ry: 20, fill: "#eef2f1", stroke: "#c6d1d4"}));
  for (const r of [71, 99, 128]) tower.append(node("ellipse", {cx: 660, cy: 285, rx: r, ry: 34 + r * .14, fill: "none", stroke: "#087c88", "stroke-width": 1, opacity: .12}));
  tower.append(node("path", {d: "M645 350 L660 198 L675 350 M651 299 H669 M653 273 H667 M655 247 H665 M658 222 H662 M637 350 H683", fill: "none", stroke: "#1a2c35", "stroke-width": 3.2, "stroke-linejoin": "round", opacity: .85}));
  tower.append(node("path", {d: "M650 315 L671 294 M671 315 L650 294 M653 282 L668 266 M668 282 L653 266 M656 250 L665 240 M665 250 L656 240", fill: "none", stroke: "#087c88", "stroke-width": 1.2, opacity: .8}));
  for (let i = 0; i < 8; i++) {
    const [ax, ay] = ANTENNAS[i];
    // Receiver noise is added at the antenna, not at the transmitting phone.
    if (frame) tower.append(node("circle", {cx: ax, cy: ay + 15, r: 4 + Math.min(7, sigma2 * 3), fill: "#a8701b", opacity: .08 + Math.min(.2, sigma2 * .1)}));
    tower.append(node("rect", {x: ax - 2.2, y: ay, width: 4.4, height: 30, rx: 1, fill: "#087c88", stroke: "none"}));
    tower.append(node("circle", {cx: ax, cy: ay - 5, r: 1.8, fill: "#075e67", opacity: .9}));
  }
  tower.append(node("rect", {x: 650, y: 351, width: 20, height: 18, rx: 1, fill: "#fbfcfa", stroke: "#087c88"}));
  tower.append(node("circle", {cx: 656, cy: 359, r: 1.8, fill: "#087c88"}), node("circle", {cx: 663, cy: 359, r: 1.8, fill: "#a8701b"}));
  tower.append(node("text", {x: 660, y: 391, "text-anchor": "middle", fill: "#52656d", "font-size": 10, "font-family": "system-ui"}, "8 antenna receiver"));
  svg.append(tower);

  for (let i = 0; i < 12; i++) {
    const [ux, uy] = usersPositions[i];
    const active = i < users, error = selectedErrors.has(i);
    const group = node("g", {transform: `translate(${ux} ${uy})`, opacity: active ? 1 : .32, class: "device-node"});
    group.setAttribute("data-user", i);
    if (active && block.has(i)) {
      group.append(node("circle", {r: 18, fill: "none", stroke: "#2563eb", "stroke-width": 2.2, "stroke-dasharray": "3 3", class: "selected-block-ring"}));
    }
    const uncertainty = Math.max(0, Math.min(1, Number(uncertainties[i] || 0)));
    if (active) group.append(node("circle", {r: 8 + 13 * uncertainty, fill: "#a8701b", opacity: .035 + .19 * uncertainty, class: "device-halo"}));
    group.append(node("rect", {x: -8, y: -12, width: 16, height: 25, rx: 2, fill: "#fbfcfa", stroke: error ? "#b44632" : (active ? "#087c88" : "#9aabb0"), "stroke-width": error ? 1.8 : 1.2}));
    group.append(node("rect", {x: -5, y: -8.2, width: 10, height: 14, rx: 1, fill: "#e8edef", stroke: "none"}));
    group.append(node("circle", {cx: 0, cy: 9.2, r: 1, fill: error ? "#b44632" : "#087c88"}));
    if (error) group.append(node("text", {x: 0, y: -19, "text-anchor": "middle", fill: "#b44632", "font-size": 10, "font-weight": 700}, "!"));
    group.append(node("text", {x: 0, y: 28, "text-anchor": "middle", fill: "#111827", "font-size": 9, "font-family": "ui-monospace,monospace"}, `U${String(i + 1).padStart(2, "0")}${active && detected[i] ? ` · ${detected[i]}` : active && truth[i] ? ` · ${truth[i]}` : ""}`));
    if (active && truth[i]) group.append(node("title", {}, `User ${i + 1}: transmitted ${truth[i]}${detected[i] ? `, detected ${detected[i]}` : ""}${error ? " (error)" : ""}`));
    svg.append(group);
  }

  if (users > 8) {
    const tag = node("g");
    tag.append(node("rect", {x: 793, y: 19, width: 172, height: 26, rx: 1, fill: "#f5e2dc", stroke: "#b44632"}));
    tag.append(node("circle", {cx: 810, cy: 32, r: 3, fill: "#b44632"}));
    tag.append(node("text", {x: 823, y: 36, fill: "#1a2c35", "font-size": 10, "font-family": "ui-monospace,monospace"}, "Too many phones · 12 / 8"));
    svg.append(tag);
  }
  if (frame) {
    svg.append(node("text", {x: 34, y: 31, fill: "#52656d", "font-size": 10, "font-family": "ui-monospace,monospace"}, `Example ${String(frame).padStart(2, "0")} · ${users} phones`));
  }
  container.replaceChildren(svg);
  setNetworkMotion(container, reducedMotion || document.body.classList.contains("motion-paused"));
  container.setAttribute("aria-label", `Schematic uplink: ${users} phones each show one signal line to the receiver antenna array${users > 8 ? ", overloaded" : ""}. ${block.size ? `Users ${[...block].map((user) => user + 1).join(", ")} form the current correction block. ` : ""}${selectedErrors.size} user detection errors.`);
}
