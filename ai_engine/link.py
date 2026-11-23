"""The link guess (M-S2a): P(device class | room type, gang count, gang index), and grouping
detected switch keys into plates. `linker/linker.py` fits and scores the same table.

It takes no settings at import: every number is passed in, so the enclave can load it with the
settings it gets over vsock (C-12)."""

import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["ceiling_light", "downlight", "pendant_light", "wall_light", "floor_lamp",
           "cabinet_light", "accent_light", "exhaust_fan", "other"]
ROOMS = ["living_room", "bedroom", "kitchen", "bathroom", "hallway", "utility_room", "dressing_room", "other"]


def position(i, n):
    if i is None or not n:
        return "unknown"
    return "only" if n == 1 else "first" if i == 0 else "last" if i == n - 1 else "middle"


# Rules the table starts from, as pseudo-counts. Written from how Korean apartments are
# usually wired; every real or verified count is added on top.
def rule_counts(room, pos):
    c = Counter({k: 0.2 for k in CLASSES})
    if pos in ("only", "first", "unknown"):
        c["ceiling_light"] += 3
    else:
        c["accent_light"] += 2
        c["downlight"] += 0.5
    if room == "bathroom" and pos in ("middle", "last"):
        c["exhaust_fan"] += 3
    if room == "kitchen" and pos in ("middle", "last"):
        c["cabinet_light"] += 1.5
    if room in ("living_room", "bedroom") and pos in ("middle", "last"):
        c["pendant_light"] += 0.5
    return c


class Prior:
    """Counts at three levels with back-off: (room, n, i) -> (room, position) -> (position)."""

    def __init__(self, alpha, floor):
        self.alpha, self.floor = alpha, floor
        self.c = {lvl: defaultdict(Counter) for lvl in ("exact", "room_pos", "pos")}

    @staticmethod
    def keys(room, n, i):
        pos = position(i, n)
        return {"exact": f"{room}|{n}|{i}", "room_pos": f"{room}|{pos}", "pos": pos}

    def add(self, room, n, i, cls, w=1.0):
        for lvl, k in self.keys(room, n, i).items():
            self.c[lvl][k][cls] += w

    def probs(self, room, n, i):
        k = self.keys(room, n, i)
        base = rule_counts(room, k["pos"])
        p = {c: base[c] / sum(base.values()) for c in CLASSES}
        for lvl in ("pos", "room_pos", "exact"):             # coarse to fine
            cnt = self.c[lvl].get(k[lvl], Counter())
            tot = sum(cnt.values())
            p = {c: (cnt[c] + self.alpha * p[c]) / (tot + self.alpha) for c in CLASSES}
        # a guess is never certain before a press
        return {c: (1 - self.floor) * p[c] + self.floor / len(CLASSES) for c in CLASSES}

    def save(self, path):
        Path(path).write_text(json.dumps({"alpha": self.alpha, "counts": {l: {k: dict(v) for k, v in d.items()} for l, d in self.c.items()}}, indent=1))

    @classmethod
    def load(cls, path, floor):
        d = json.loads(Path(path).read_text())
        p = cls(d["alpha"], floor)
        for lvl, table in d["counts"].items():
            for k, v in table.items():
                p.c[lvl][k] = Counter(v)
        return p


def plates(boxes, plate_gap):
    """Group switch keys that touch into plates and number them column by column from the left.

    `boxes` are `(kind, [x1, y1, x2, y2])`. Returns one `(count, index)` per box, or None for a
    box that is not a switch key."""
    xywh = [(x1, y1, x2 - x1, y2 - y1) for _, (x1, y1, x2, y2) in boxes]
    idx = [i for i, (kind, _) in enumerate(boxes) if kind == "switch"]
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
            (x0, y0, w0, h0), (x1, y1, w1, h1) = xywh[a], xywh[b]
            gx = max(x0, x1) - min(x0 + w0, x1 + w1)          # gap, negative when overlapping
            gy = max(y0, y1) - min(y0 + h0, y1 + h1)
            side = gx < plate_gap * max(w0, w1) and gy < -0.5 * min(h0, h1)
            stacked = gy < plate_gap * max(h0, h1) and gx < -0.5 * min(w0, w1)
            if side or stacked:
                parent[find(a)] = find(b)
    groups = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    out = [None] * len(boxes)
    for members in groups.values():
        w = max(min(xywh[i][2] for i in members), 1e-6)
        members.sort(key=lambda i: (round(xywh[i][0] / w), xywh[i][1]))
        for n, i in enumerate(members):
            out[i] = (len(members), n)
    return out
