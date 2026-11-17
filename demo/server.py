"""Camera demo for the control detector.

    python demo/server.py [--weights models/detector-yolo11n-640-synth.pt] [--port 8000]

Serves index.html and POST /detect (body: one JPEG frame). The weights file is reloaded
whenever it changes on disk, so a model that is still training shows up as it improves.
"""

import argparse
import base64
import csv
import io
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "linker", ROOT / "effect_check"):
    sys.path.insert(0, str(p))
import numpy as np   # noqa: E402
from linker import Prior   # noqa: E402
from settings import S, path   # noqa: E402

SPOT = S["effect_check"]["spot"]
PLATE_GAP = S["linker"]["plate_gap"]

HERE = Path(__file__).parent
NAMES_KO = {"switch": "스위치", "toggle": "토글", "power_strip": "멀티탭"}
DEVICE_KO = {"ceiling_light": "천장등", "downlight": "다운라이트", "pendant_light": "펜던트등", "wall_light": "벽등",
             "floor_lamp": "스탠드", "cabinet_light": "주방 상부등", "accent_light": "보조등", "exhaust_fan": "환풍기",
             "other": "기타"}
PRIOR_PATH = path("prior")
PRIOR = Prior.load(PRIOR_PATH) if PRIOR_PATH.exists() else Prior()


def plates(boxes):
    """Group switch keys that touch into plates; number them column by column from the left."""
    idx = [i for i, b in enumerate(boxes) if b["kind"] == "switch"]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in idx:
        for b in idx:
            if a >= b:
                continue
            (x0, y0, w0, h0), (x1, y1, w1, h1) = boxes[a]["box"], boxes[b]["box"]
            gx = max(x0, x1) - min(x0 + w0, x1 + w1)          # gap, negative when overlapping
            gy = max(y0, y1) - min(y0 + h0, y1 + h1)
            side = gx < PLATE_GAP * max(w0, w1) and gy < -0.5 * min(h0, h1)
            stacked = gy < PLATE_GAP * max(h0, h1) and gx < -0.5 * min(w0, w1)
            if side or stacked:
                parent[find(a)] = find(b)
    groups = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    for members in groups.values():
        w = min(boxes[i]["box"][2] for i in members)
        members.sort(key=lambda i: (round(boxes[i]["box"][0] / w), boxes[i]["box"][1]))
        for n, i in enumerate(members):
            boxes[i]["gang_index"], boxes[i]["gang_count"] = n, len(members)


def guess_links(boxes, room):
    plates(boxes)
    for b in boxes:
        if b["kind"] != "switch":
            continue
        p = PRIOR.probs(room, b["gang_count"], b["gang_index"])
        top = sorted(p.items(), key=lambda kv: -kv[1])[:2]
        b["guess"] = [{"device": d, "label": DEVICE_KO[d], "p": round(v, 2)} for d, v in top]

def changed_spot(before, after):
    """Box around the brightest spot that lit up (or went dark): a lamp that changed in view.
    Returns [x, y, w, h] ratios or None."""
    size = (96, 72)
    b = np.asarray(before.convert("L").resize(size), dtype=np.float32) / 255
    a = np.asarray(after.convert("L").resize(size), dtype=np.float32) / 255
    gain = np.median(b[b > 0.05]) / max(np.median(a[a > 0.05]), 1e-3) if (a > 0.05).any() and (b > 0.05).any() else 1
    d = a * min(max(gain, 0.8), 1.25) - b
    best = None
    for sign, lit in ((1, a), (-1, b)):                 # turned on: bright after; turned off: bright before
        dd = d * sign
        if dd.max() < SPOT["min_rise"]:
            continue
        mask = (dd > max(SPOT["min_rise"], 0.5 * dd.max())) & (lit > SPOT["min_lit"])
        if mask.sum() < 3:
            continue
        # largest connected blob
        seen, comps = np.zeros_like(mask), []
        for y0, x0 in zip(*np.nonzero(mask)):
            if seen[y0, x0]:
                continue
            stack, pts = [(y0, x0)], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < size[1] and 0 <= nx < size[0] and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(pts)
        pts = max(comps, key=len)
        score = sum(dd[y, x] for y, x in pts)
        if best is None or score > best[0]:
            ys, xs = [p[0] for p in pts], [p[1] for p in pts]
            best = (score, [min(xs) / size[0], min(ys) / size[1], (max(xs) - min(xs) + 1) / size[0], (max(ys) - min(ys) + 1) / size[1]])
    return best[1] if best else None


def check(before, after):
    """Stage 3 on two photos: what did the press change?"""
    from learned import compare
    spot = changed_spot(before, after)
    r = compare(before, after, [{"name": "spot", "bbox": spot}] if spot else [])
    if r["device"]:
        kind, text = "in_view", ("화면 안 조명이 켜짐" if r["change"] == "off_to_on" else "화면 안 조명이 꺼짐")
    elif r["changed"]:
        kind, text = "offscreen", ("화면 밖 조명이 켜짐" if r["change"] == "off_to_on" else "화면 밖 조명이 꺼짐")
    else:
        kind, text = "none", "화면에 변화 없음 (환풍기나 다른 방 장치일 수 있음)"
    return {"result": kind, "text": text, "change": r["change"], "confidence": r["confidence"],
            "box": spot if kind == "in_view" else None}


ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(path("detector_weights")))
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--device", default="cpu")
A = ap.parse_args()
WEIGHTS = Path(A.weights)

lock = threading.Lock()
state = {"model": None, "mtime": None}


def model():
    if not WEIGHTS.exists():
        return None
    m = WEIGHTS.stat().st_mtime
    if m != state["mtime"]:
        try:
            state["model"], state["mtime"] = YOLO(str(WEIGHTS)), m
        except Exception:          # file caught mid-write: keep the old model
            pass
    return state["model"]


def info():
    out = {"weights": WEIGHTS.name, "ready": WEIGHTS.exists()}
    results = WEIGHTS.parent.parent / "results.csv"
    if results.exists():
        rows = list(csv.DictReader(results.open()))
        if rows:
            best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
            out.update(epoch=int(float(rows[-1]["epoch"])), best_epoch=int(float(best["epoch"])),
                       map50=round(float(best["metrics/mAP50(B)"]), 3))
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/info":
            return self.send_json(info())
        return super().do_GET()

    def do_POST(self):
        if self.path == "/check":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            imgs = [Image.open(io.BytesIO(base64.b64decode(body[k].split(",", 1)[1]))).convert("RGB") for k in ("before", "after")]
            with lock:
                return self.send_json(check(*imgs))
        if self.path != "/detect":
            return self.send_json({"error": "not found"}, 404)
        conf = float(self.headers.get("X-Conf", "0.3"))
        img = Image.open(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"])))).convert("RGB")
        with lock:
            m = model()
            if m is None:
                return self.send_json({"ready": False, "boxes": []})
            r = m.predict(img, imgsz=640, conf=conf, device=A.device, verbose=False)[0]
        W, H = img.size
        boxes = [{"kind": r.names[int(c)], "label": NAMES_KO.get(r.names[int(c)], r.names[int(c)]),
                  "conf": round(float(s), 2), "box": [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]}
                 for (x0, y0, x1, y1), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        guess_links(boxes, self.headers.get("X-Room", "living_room"))
        self.send_json({"ready": True, "boxes": boxes, "ms": round(sum(r.speed.values()))})


if __name__ == "__main__":
    print(f"http://localhost:{A.port}  weights={WEIGHTS}")
    ThreadingHTTPServer(("127.0.0.1", A.port), Handler).serve_forever()
