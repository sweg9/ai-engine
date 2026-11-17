"""Did pressing a key change something, and which lamp? Compares a before and an after photo.

Usage:
    python3 change_check.py --before a.jpg --after b.jpg [--devices devices.json]
    python3 change_check.py --eval evidence      # score on the synthetic before and after pairs

First stage for FR-02: where the picture got brighter or darker. With lamp boxes it also
names the lamp whose box changed most. A mapping becomes Verified only when this check
and the VLM agree; this script does not ask the VLM.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import S   # noqa: E402

MIN_CHANGE = S["effect_check"]["simple"]["min_change"]   # mean luminance change of a grid cell, 0 to 1
MIN_SHARE = S["effect_check"]["simple"]["min_share"]     # share of cells that must change
GRID = 16


def luminance(path, size=(320, 240)):
    return np.asarray(Image.open(path).convert("L").resize(size), dtype=np.float32) / 255


def compare(before, after, devices=None):
    a, b = luminance(before), luminance(after)
    # a phone re-exposes a little between shots: remove a global gain before comparing
    gain = np.median(a[a > 0.05]) / max(np.median(b[b > 0.05]), 1e-3) if (a > 0.05).any() and (b > 0.05).any() else 1
    d = b * min(max(gain, 0.8), 1.25) - a
    h, w = d.shape
    cells = d[: h // GRID * GRID, : w // GRID * GRID].reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3))
    changed = np.abs(cells) >= MIN_CHANGE
    share = float(changed.mean())
    if share < MIN_SHARE:
        return {"changed": False, "change": None, "device": None, "confidence": round(1 - share / MIN_SHARE, 2),
                "reason": "no part of the picture changed enough"}
    mean = float(cells[changed].mean())
    out = {"changed": True, "change": "off_to_on" if mean > 0 else "on_to_off",
           "confidence": round(min(1.0, abs(mean) / MIN_CHANGE * 0.3 + share), 2),
           "reason": f"{share:.0%} of the picture got {'brighter' if mean > 0 else 'darker'}", "device": None}
    if devices:
        best, score = None, 0.0
        for dv in devices:
            if not dv.get("bbox"):
                continue
            x, y, bw, bh = dv["bbox"]
            x0, y0 = int(max(0, x - bw * 0.2) * w), int(max(0, y - bh * 0.2) * h)
            x1, y1 = int(min(1, x + bw * 1.2) * w) + 1, int(min(1, y + bh * 1.2) * h) + 1
            s = float(np.abs(d[y0:y1, x0:x1]).mean())
            if s > score:
                best, score = dv["name"], s
        if score >= MIN_CHANGE * 2:
            out["device"] = best
            out["reason"] += f"; the change is strongest on {best}"
        else:
            out["reason"] += "; no lamp in view changed, so the lamp is out of view"
    return out


def evaluate(root):
    pairs = []
    for labels in Path(root).rglob("labels.jsonl"):
        folder = labels.parent
        recs = {json.loads(l)["image_id"]: json.loads(l) for l in labels.read_text().splitlines() if l.strip()}
        for r in recs.values():
            for c in r["controls"]:
                ev = c.get("evidence")
                if ev and ev["before_image"] == r["image_id"]:
                    devs = r["source"]["devices"]
                    visible = any(d["name"] == ev["changed_device"] and d["bbox"] for d in devs)
                    pairs.append((folder / "images" / f"{ev['before_image']}.jpg", folder / "images" / f"{ev['after_image']}.jpg",
                                  ev, devs, visible))
    stats = {"pairs": len(pairs)}
    lamp = [p for p in pairs if p[2]["changed_device"] != "exhaust_fan"]
    fan = [p for p in pairs if p[2]["changed_device"] == "exhaust_fan"]
    res = {id(p): compare(p[0], p[1], p[3]) for p in pairs}
    stats["lamp_pairs"] = len(lamp)
    stats["lamp_change_found"] = round(sum(res[id(p)]["changed"] for p in lamp) / max(len(lamp), 1), 3)
    stats["lamp_direction_right"] = round(sum(res[id(p)]["change"] == p[2]["change"] for p in lamp) / max(len(lamp), 1), 3)
    vis = [p for p in lamp if p[4]]
    stats["lamp_in_view_pairs"] = len(vis)
    stats["lamp_in_view_named_right"] = round(sum(res[id(p)]["device"] == p[2]["changed_device"] for p in vis) / max(len(vis), 1), 3)
    stats["fan_pairs"] = len(fan)
    stats["fan_false_alarm"] = round(sum(res[id(p)]["changed"] for p in fan) / max(len(fan), 1), 3)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--devices")
    ap.add_argument("--eval")
    a = ap.parse_args()
    if a.eval:
        evaluate(a.eval)
    else:
        devs = json.loads(Path(a.devices).read_text()) if a.devices else None
        print(json.dumps(compare(a.before, a.after, devs), ensure_ascii=False))
