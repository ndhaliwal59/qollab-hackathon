"""Local, precomputed demo and plots; no network requests or large browser data."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def make_artifacts(run_dir: Path, report, replays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ("mmse", "mmse_sic", "classical", "uniform", "biased", "sa", "qaoa", "ml_reference")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, loading in zip(axes, ("square", "overloaded")):
        snrs = sorted({cell["snr_db"] for cell in report["cells"]})
        for name in methods:
            ber = [np.mean([cell["summary"][name]["ber"] for cell in report["cells"]
                            if cell["loading"] == loading and cell["csi_error_db"] == "perfect"
                            and cell["snr_db"] == snr]) for snr in snrs]
            ax.plot(snrs, ber, marker="o", markersize=3, label=name,
                    linewidth=2.2 if name == "qaoa" else 1)
        ax.set(title=f"{loading}: perfect CSI, declared families pooled", xlabel="SNR (dB)", ylabel="Bit error rate (errors/bit)")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.suptitle("Held-out complex BPSK detection · local finite-shot simulation")
    fig.tight_layout()
    fig.savefig(run_dir / "ber_vs_snr.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in methods[:-1]:
        summary = report["primary_summary"][name]
        x, y = summary["latency_s"]["p95"] * 1000, summary["ber"]
        ax.scatter(x, y, s=45 if name == "qaoa" else 20, label=name)
    ax.set(xscale="log", xlabel="Warm per-detection p95 latency (ms, log scale)", ylabel="Primary BER (errors/bit)",
           title="Quality vs local simulator latency · SNR 8/12/16/20 dB, perfect CSI")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right", fontsize=8, title="Detector")
    fig.tight_layout()
    fig.savefig(run_dir / "quality_latency.png", dpi=160)
    plt.close(fig)
    names = [name for name in ("qaoa", "opposite_warm", "uncertain_only", "no_refinement", "real_only")
             if name in report["primary_summary"]]
    if len(names) > 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, [report["primary_summary"][name]["ber"] for name in names], color="0.4")
        ax.set(ylabel="Primary BER (errors/bit)", title="Frozen ablations · perfect CSI, SNR 8/12/16/20 dB")
        ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "ablations.png", dpi=160)
        plt.close(fig)
    payload = json.dumps({"conclusion": report["conclusion"], "profile": report["settings"]["profile"],
                          "summary": report["primary_summary"], "gates": report["success_gates"],
                          "replays": replays, "freeze": report["freeze_sha256"]}).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Signal Rescue · hybrid lab</title>
<style>body{font:15px system-ui,sans-serif;background:#f5f4ef;color:#172923;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:30px;margin:8px 0}h2{font-size:20px;margin-top:32px}.label{letter-spacing:.14em;text-transform:uppercase;font-size:11px}p{line-height:1.6;max-width:850px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{font-size:12px;color:#4b5b52}section{margin:24px 0}button,select{font:inherit;padding:9px 14px;border:1px solid #aaa;background:white;border-radius:4px}button{cursor:pointer}pre{overflow:auto;background:#e6ebe6;padding:16px;font:13px ui-monospace,monospace}.note{font-size:12px;color:#53635a}.status{padding:16px;border-left:4px solid #3d6756;background:#e4ebe6}img{max-width:100%}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.plots{grid-template-columns:1fr}main{padding:16px}td,th{padding:7px}}</style>
<main><div class="label">5G uplink / reproducible experiment</div><h1>Signal Rescue: the hybrid lab</h1>
<p>MMSE → bounded single/pair search → uncertain coupled bits → finite-shot QAOA → common refinement.
Classical samplers use the same candidate and refinement budgets.</p>
<p id="status" class="status"></p><p class="note" id="profile"></p>
<h2>Held-out primary results</h2><p class="note">Fixed SNRs 8, 12, 16, 20 dB · perfect CSI · declared channel families pooled. Latency is local simulation, not radio real-time or QPU latency.</p>
<table><thead><tr><th>Detector</th><th>BER</th><th>BLER</th><th>p95 latency (ms)</th></tr></thead><tbody id="results"></tbody></table>
<section><h2>One bounded, precomputed detection</h2><p>Replay actual sampled candidates and the accept/reject decisions. No network calls or new optimization run in this page.</p>
<select id="sample" aria-label="Choose channel sample"></select> <button id="advance">Next correction round</button> <button id="reset">Reset replay</button><pre id="replay"></pre></section>
<h2>Benchmark gates</h2><ul id="gates"></ul>
<div class="plots"><img src="ber_vs_snr.png" alt="BER versus SNR for square and overloaded channels"><img src="quality_latency.png" alt="Primary BER versus local simulator p95 latency"></div>
<p class="note">DeepMIMO is ray-traced CSI, with synthetic BPSK symbols and circular complex noise. Controlled models are labeled separately in report.json.
The old mMobile axis-remapping reports are excluded. A lower residual does not guarantee fewer transmitted-bit errors; beating MMSE alone is not quantum advantage.</p>
<p class="note" id="freeze"></p></main><script>
const data=__DATA__;
document.getElementById('status').textContent=data.conclusion;
document.getElementById('profile').textContent='Run profile: '+data.profile+' · all values are precomputed from the frozen held-out run.';
document.getElementById('freeze').textContent='Frozen configuration SHA-256: '+data.freeze;
const body=document.getElementById('results');
for(const [name,r] of Object.entries(data.summary)){const tr=document.createElement('tr');for(const text of [name,r.ber.toFixed(5),r.bler.toFixed(5),(1000*r.latency_s.p95).toFixed(2)]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td)}body.appendChild(tr)}
for(const [name,pass] of Object.entries(data.gates)){const li=document.createElement('li');li.textContent=(pass?'PASS · ':'NOT MET · ')+name.replaceAll('_',' ');document.getElementById('gates').appendChild(li)}
const picker=document.getElementById('sample');let round=0;
data.replays.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.family+' / '+r.loading+' / '+r.snr_db+' dB';picker.appendChild(o)});
function render(){const r=data.replays[Number(picker.value)||0];if(!r){document.getElementById('replay').textContent='This run did not include replay samples.';return}const shown={channel:r.family,loading:r.loading,truth:r.truth,mmse_bits:r.mmse_bits,classical_incumbent:r.classical_incumbent,completed_rounds:r.rounds.slice(0,round)};if(round>=r.rounds.length){shown.final_bits=r.final_bits;shown.note=r.note}document.getElementById('replay').textContent=JSON.stringify(shown,null,2);document.getElementById('advance').disabled=round>=r.rounds.length}
picker.onchange=()=>{round=0;render()};document.getElementById('advance').onclick=()=>{round++;render()};document.getElementById('reset').onclick=()=>{round=0;render()};render();
</script></html>"""
    (run_dir / "demo.html").write_text(html.replace("__DATA__", payload))
