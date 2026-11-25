"""Render the demo bedroom set on this machine. The images never go to git (C-05).

    python synthetic/render.py            # renders the seed folders that are missing
    python synthetic/render.py --force    # renders every seed again

detector/build_yolo.py runs this first, so a fresh clone only needs Blender (4.5 or later).
Seeds, scene count and flags are in settings.yaml (`synthetic`), the output folder is
`paths.demo_root`. The same seed gives the same scenes and labels on every machine.

Blender is looked up in this order: `synthetic.blender` in settings.yaml, the BLENDER
environment variable, `blender` on PATH, the macOS app.
"""

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import S, path   # noqa: E402

MAC_APP = "/Applications/Blender.app/Contents/MacOS/Blender"


def blender():
    for c in (S["synthetic"].get("blender"), os.environ.get("BLENDER"), shutil.which("blender"), MAC_APP):
        if c and Path(c).is_file():
            return c
    sys.exit("Blender not found. Install Blender 4.5 or later, then set synthetic.blender in settings.yaml "
             "or the BLENDER environment variable. To build without the demo set: build_yolo.py --no-render")


def seeds():
    first, last = S["synthetic"]["demo"]["seeds"]
    return range(first, last + 1)


def missing():
    out = path("demo_root")
    return [s for s in seeds() if not (out / f"syn-e{s}" / "labels.jsonl").exists()]


def render_one(exe, seed):
    """Render one seed into a temporary folder and move it into place only when Blender finished,
    so a stopped run never leaves a half folder that looks complete."""
    d = S["synthetic"]["demo"]
    out = path("demo_root")
    tmp, final = out / f"syn-e{seed}.part", out / f"syn-e{seed}"
    shutil.rmtree(tmp, ignore_errors=True)
    log = out / f"log_e{seed}.txt"
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([exe, "-b", "--factory-startup", "-P", str(ROOT / "synthetic" / "synth_evidence.py"), "--",
                            "--out", str(tmp), "--scenes", str(d["scenes"]), "--seed", str(seed),
                            *(str(a) for a in d["args"])],
                           stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=False)
    if r.returncode != 0 or not (tmp / "labels.jsonl").exists():
        return f"syn-e{seed}: failed, see {log}"
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    n = len((final / "labels.jsonl").read_text(encoding="utf-8").splitlines())
    return f"syn-e{seed}: {n} images"


def render(todo):
    if not todo:
        return True
    exe = blender()
    path("demo_root").mkdir(parents=True, exist_ok=True)
    print(f"rendering demo seeds {todo} into {path('demo_root')} with {exe}", flush=True)
    with ThreadPoolExecutor(S["synthetic"]["parallel"]) as pool:
        results = list(pool.map(lambda s: render_one(exe, s), todo))
    for line in results:
        print(line, flush=True)
    return not any("failed" in line for line in results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="render every demo seed again")
    a = ap.parse_args()
    sys.exit(0 if render(list(seeds()) if a.force else missing()) else 1)


if __name__ == "__main__":
    main()
