"""Build the explicit, size-checked browser handoff under dist/qollab."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "qollab"
PYTHON_FILES = ("__init__.py", "core.py", "channels.py", "hybrid.py", "samplers.py", "circuits.py", "live.py")
ASSETS = (
    "data/deepmimo/asu_slice.npz",
    "data/deepmimo/asu_slice.json",
    "results/confirmation-full/frozen.json",
    "results/confirmation-full/report.json",
)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "signal_rescue").mkdir()
    for name in PYTHON_FILES:
        shutil.copy2(ROOT / "signal_rescue" / name, OUT / "signal_rescue" / name)
    for name in ASSETS:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(ROOT / "web" / "index.html", OUT / "index.html")
    shutil.copy2(ROOT / "web" / "main.css", OUT / "main.css")
    shutil.copy2(ROOT / "web" / "main.js", OUT / "main.js")
    for folder in ("runtime", "ui"):
        shutil.copytree(ROOT / "web" / folder, OUT / folder)
    manifest = []
    violations = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        relative = path.relative_to(OUT).as_posix()
        size = path.stat().st_size
        manifest.append({"path": relative, "bytes": size})
        if size > 2_000_000:
            violations.append(f"{relative}: {size} bytes exceeds 2 MB")
    if len(manifest) + 1 > 100:
        violations.append(f"{len(manifest) + 1} files including the manifest exceeds the 100-file project limit")
    (OUT / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": manifest, "count": len(manifest)}, indent=2) + "\n")
    if violations:
        raise SystemExit("Qollab package limits failed:\n" + "\n".join(violations))
    print(f"Packaged {len(manifest)} project files plus manifest ({sum(item['bytes'] for item in manifest):,} payload bytes) in {OUT}")
    print("Every file is under 2 MB; total file count is under 100.")


if __name__ == "__main__":
    main()
