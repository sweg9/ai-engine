"""Stage 3: a learned before/after check. Which lamp did this press turn on or off?

    python effect_check/learned.py --train      # cache images, train, save the weights in settings
    python effect_check/learned.py --eval       # score on the held-out seeds

Same answer shape as effect_check/change_check.py `compare`, so it can replace it:
    {"changed": bool, "change": "off_to_on" | "on_to_off" | None, "device": name | None, ...}

One small siamese network (MobileNetV3-small, ImageNet weights, shared between the two photos)
with two heads, each giving none / turned on / turned off:
- a scene head on the whole picture: did anything in view change?
- a lamp head on a crop around each lamp in view: did this lamp change?
Each head also gets four plain brightness numbers of its region (mean change, mean absolute
change, 95th and 5th percentile), so a faint change is not lost in the features.

Training data are the synthetic before/after pairs. Extra "nothing changed" pairs are made by
pairing a photo with a re-exposed, slightly shifted copy of itself, as a phone does between
two shots. Held out: the last two seed folders.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import S, path   # noqa: E402

E = S["effect_check"]
CACHE = path("effect_cache")
WEIGHTS = path("effect_weights")
SCENE = (192, 144)
CROP = 96
LABELS = ["none", "off_to_on", "on_to_off"]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------- data

def pairs(root):
    """(folder, before id, after id, changed device, change, devices of the before shot)."""
    for labels in sorted(Path(root).glob("syn-e*/labels.jsonl")):
        for line in labels.read_text().splitlines():
            r = json.loads(line)
            for c in r["controls"]:
                ev = c.get("evidence")
                if ev and ev["before_image"] == r["image_id"]:
                    yield labels.parent, ev["before_image"], ev["after_image"], ev["changed_device"], ev["change"], r["source"]["devices"]


def cached(folder, image_id, size=(384, 288)):
    p = CACHE / folder.name / f"{image_id}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.open(folder / "images" / f"{image_id}.jpg").convert("RGB").resize(size, Image.BILINEAR).save(p, quality=95)
    return p


def to_tensor(im):
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)


def stats(b, a):
    d = (a - b).mean(0).flatten()
    return torch.stack([d.mean(), d.abs().mean(), d.quantile(0.95), d.quantile(0.05)]) * 10


def crop_box(bbox, W, H):
    x, y, w, h = bbox
    cx, cy, s = (x + w / 2) * W, (y + h / 2) * H, max(w * W, h * H) * 1.6 + 8
    return int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)


def views(before, after, devices, jitter=None):
    """Scene tensors, lamp-crop tensors and stats for one pair of PIL images."""
    if jitter:
        after = jitter(after)
    W, H = before.size
    sb, sa = to_tensor(before.resize(SCENE)), to_tensor(after.resize(SCENE))
    out = {"scene": (sb, sa, stats(sb, sa)), "lamps": []}
    for d in devices:
        if not d.get("bbox"):
            continue
        box = crop_box(d["bbox"], W, H)
        cb, ca = to_tensor(before.crop(box).resize((CROP, CROP))), to_tensor(after.crop(box).resize((CROP, CROP)))
        out["lamps"].append((d["name"], cb, ca, stats(cb, ca)))
    return out


def phone_jitter(rng):
    def f(im):
        W, H = im.size
        dx, dy = rng.randint(-int(W * 0.02), int(W * 0.02)), rng.randint(-int(H * 0.02), int(H * 0.02))
        im = im.transform(im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) * rng.uniform(0.85, 1.15) + np.random.normal(0, 2, (H, W, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return f


def half(ts):
    return tuple(t.half() for t in ts)       # keeps the sample cache at about half the memory


def build_samples(root, folders, rng, same_shot_share=0.35):
    """Lists of (before, after, stats, label) for the scene head and the lamp head."""
    scene, lamp = [], []
    for folder, b, a, dev, change, devices in pairs(root):
        if folder.name not in folders:
            continue
        B, A = Image.open(cached(folder, b)), Image.open(cached(folder, a))
        lab = 0 if dev == "exhaust_fan" else LABELS.index(change)
        v = views(B, A, devices, phone_jitter(rng))
        scene.append((*half(v["scene"]), lab))
        for name, cb, ca, st in v["lamps"]:
            lamp.append((*half((cb, ca, st)), LABELS.index(change) if name == dev else 0))
        if rng.random() < same_shot_share:           # nothing changed: the same shot, re-exposed and shifted
            v = views(B, B, devices, phone_jitter(rng))
            scene.append((*half(v["scene"]), 0))
            for name, cb, ca, st in v["lamps"]:
                lamp.append((*half((cb, ca, st)), 0))
    return scene, lamp


# ---------- model

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.f = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        head = lambda: nn.Sequential(nn.Linear(576 * 3 + 4, 192), nn.ReLU(), nn.Dropout(0.2), nn.Linear(192, 3))
        self.scene, self.lamp = head(), head()

    def forward(self, b, a, st, which):
        fb, fa = self.f((b - MEAN.to(b)) / STD.to(b)), self.f((a - MEAN.to(a)) / STD.to(a))
        z = torch.cat([fa - fb, fa, fb, st], 1)
        return (self.scene if which == "scene" else self.lamp)(z)


def batches(samples, size, shuffle=True):
    idx = list(range(len(samples)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), size):
        chunk = [samples[j] for j in idx[i:i + size]]
        yield [torch.stack([c[k] for c in chunk]).float() for k in range(3)] + [torch.tensor([c[3] for c in chunk])]


def weights(samples):
    n = np.bincount([s[3] for s in samples], minlength=3) + 1
    return torch.tensor((n.sum() / n / 3) ** 0.5, dtype=torch.float32)


def split(root):
    folders = sorted(p.name for p in Path(root).glob("syn-e*"))
    k = E["held_out_folders"]
    return set(folders[:-k]), set(folders[-k:])


def train(root, epochs=E["epochs"]):
    rng = random.Random(0)
    random.seed(0)
    torch.manual_seed(0)
    tr, va = split(root)
    print("building samples…", flush=True)
    s_tr, l_tr = build_samples(root, tr, rng)
    s_va, l_va = build_samples(root, va, random.Random(1))
    print(f"scene {len(s_tr)}/{len(s_va)}  lamp {len(l_tr)}/{len(l_va)}", flush=True)
    model = Siamese().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ws, wl = weights(s_tr).to(DEV), weights(l_tr).to(DEV)
    for ep in range(epochs):
        model.train()
        for which, data, w in (("scene", s_tr, ws), ("lamp", l_tr, wl)):
            for b, a, st, y in batches(data, 32):
                loss = F.cross_entropy(model(b.to(DEV), a.to(DEV), st.to(DEV), which), y.to(DEV), weight=w)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        print(f"epoch {ep + 1}: scene acc {accuracy(model, s_va, 'scene'):.3f}  lamp acc {accuracy(model, l_va, 'lamp'):.3f}", flush=True)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)


@torch.no_grad()
def accuracy(model, samples, which):
    hit = 0
    for b, a, st, y in batches(samples, 64, shuffle=False):
        hit += (model(b.to(DEV), a.to(DEV), st.to(DEV), which).argmax(1).cpu() == y).sum().item()
    return hit / max(len(samples), 1)


# ---------- use

_model = None


@torch.no_grad()
def compare(before, after, devices=None, lamp_min=E["lamp_min"], scene_min=E["scene_min"]):
    """Drop-in for change_check.compare: before and after are paths or PIL images."""
    global _model
    if _model is None:
        _model = Siamese().to(DEV)
        _model.load_state_dict(torch.load(WEIGHTS, map_location=DEV))
        _model.eval()
    B = before if isinstance(before, Image.Image) else Image.open(before).convert("RGB")
    A = after if isinstance(after, Image.Image) else Image.open(after).convert("RGB")
    B, A = B.resize((384, 288)), A.resize((384, 288))
    v = views(B, A, devices or [])
    sb, sa, st = v["scene"]
    ps = F.softmax(_model(sb[None].to(DEV), sa[None].to(DEV), st[None].to(DEV), "scene"), 1)[0].cpu()
    best, bp, bdir = None, 0.0, None
    if v["lamps"]:
        cb = torch.stack([l[1] for l in v["lamps"]]).to(DEV)
        ca = torch.stack([l[2] for l in v["lamps"]]).to(DEV)
        cs = torch.stack([l[3] for l in v["lamps"]]).to(DEV)
        pl = F.softmax(_model(cb, ca, cs, "lamp"), 1).cpu()
        j = int((1 - pl[:, 0]).argmax())
        best, bp, bdir = v["lamps"][j][0], float(1 - pl[j, 0]), LABELS[int(pl[j, 1:].argmax()) + 1]
    if bp >= lamp_min:
        return {"changed": True, "change": bdir, "device": best, "confidence": round(bp, 2),
                "reason": f"{best} changed"}
    if 1 - ps[0] >= scene_min:
        return {"changed": True, "change": LABELS[int(ps[1:].argmax()) + 1], "device": None,
                "confidence": round(float(1 - ps[0]), 2), "reason": "the room changed but no lamp in view did"}
    return {"changed": False, "change": None, "device": None, "confidence": round(float(ps[0]), 2),
            "reason": "nothing in view changed"}


def evaluate(root):
    _, va = split(root)
    from collections import Counter
    t = Counter()
    for folder, b, a, dev, change, devices in pairs(root):
        if folder.name not in va:
            continue
        r = compare(cached(folder, b), cached(folder, a), devices)
        names = {d["name"] for d in devices}
        truth = "fan" if dev not in names else "lamp_in_view" if any(d["name"] == dev and d["bbox"] for d in devices) else "lamp_offscreen"
        obs = "no_change" if not r["changed"] else "offscreen" if not r["device"] else "right_lamp" if r["device"] == dev else "wrong_lamp"
        t[(truth, obs)] += 1
        t["direction_right"] += r["changed"] and r["change"] == change and dev != "exhaust_fan"
        t["lamp_pairs"] += dev != "exhaust_fan"
    for k in sorted(k for k in t if isinstance(k, tuple)):
        print(k, t[k])
    print("direction right on lamp pairs:", round(t["direction_right"] / max(t["lamp_pairs"], 1), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--evidence", default=str(path("synthetic_root")))
    ap.add_argument("--epochs", type=int, default=E["epochs"])
    a = ap.parse_args()
    if a.train:
        train(a.evidence, a.epochs)
    if a.eval:
        evaluate(a.evidence)
