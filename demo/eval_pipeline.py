"""Score the running demo end to end on the team eval photos (internal folders).

    python demo/eval_pipeline.py [--url http://localhost:8000]     # with demo/server.py running

For every labelled switch key: was it detected (IoU >= 0.3), did plate grouping give the right
gang count and index, and does the link guess agree with the team's role label.
"""

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "linker"))
from linker import ACCENT_GROUP, device_class   # noqa: E402
from settings import path   # noqa: E402


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(a[0], b[0])), max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(path("dataset_root")))
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    st = Counter()
    for labels in sorted(Path(a.dataset).glob("internal/*/labels.jsonl")):
        for line in labels.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "eval":
                continue
            img = next(iter((labels.parent / "images").glob(r["image_id"] + ".*")), None)
            keys = [c for c in r.get("controls") or [] if c.get("kind") == "switch" and c.get("bbox")]
            if img is None or not keys:
                continue
            req = urllib.request.Request(a.url + "/detect", data=img.read_bytes(),
                                         headers={"X-Room": r.get("room_type") or "other", "X-Conf": "0.3"})
            boxes = [b for b in json.load(urllib.request.urlopen(req))["boxes"] if b["kind"] == "switch"]
            for k in keys:
                st["keys"] += 1
                m = max(boxes, key=lambda b: iou(b["box"], k["bbox"]), default=None)
                if not m or iou(m["box"], k["bbox"]) < 0.3:
                    continue
                st["found"] += 1
                if k.get("gang_count") is not None:
                    st["gang_known"] += 1
                    st["gang_count_right"] += m.get("gang_count") == k["gang_count"]
                    st["gang_index_right"] += m.get("gang_index") == k["gang_index"]
                role = k.get("role") or {}
                if role.get("device_id") and m.get("guess"):
                    cls, g = device_class(role["device_id"]), m["guess"][0]["device"]
                    st["role_known"] += 1
                    st["link_agrees"] += g == cls or (cls == "accent_light" and g in ACCENT_GROUP)
    print(json.dumps({
        "keys": st["keys"], "found": round(st["found"] / max(st["keys"], 1), 3),
        "gang_count_right": round(st["gang_count_right"] / max(st["gang_known"], 1), 3),
        "gang_index_right": round(st["gang_index_right"] / max(st["gang_known"], 1), 3),
        "link_agrees_with_team_label": round(st["link_agrees"] / max(st["role_known"], 1), 3),
        "role_known": st["role_known"]}, indent=1))


if __name__ == "__main__":
    main()
