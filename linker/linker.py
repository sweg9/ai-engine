"""Stage 2: guess which device each detected switch key runs, before anyone presses it.

There is no real ground truth for links yet: the team `role` labels are guesses, and the
synthetic wiring is random. So the guess is a small probability table, not a trained network:

    P(device class | room type, gang count, gang index)

It starts from a few rules, takes counts from the team labels, and is meant to take counts
from verified links later (`Prior.add`). Each home then gets one link per key by assigning
the room's devices to its keys so the total probability is highest (one device per key,
except "out of view", which several keys may share).

`HomeLinker.observe` is the hook for stage 3: a press that showed a device change fixes that
link and re-assigns the rest.

    python linker/linker.py --fit      # build linker/prior.json from the team revise split
    python linker/linker.py --eval     # score it on the team eval split
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import S, path   # noqa: E402

L = S["linker"]

CLASSES = ["ceiling_light", "downlight", "pendant_light", "wall_light", "floor_lamp",
           "cabinet_light", "accent_light", "exhaust_fan", "other"]
# a guess of "accent_light" (an indirect or extra light) fits any of these
ACCENT_GROUP = {"downlight", "pendant_light", "wall_light", "floor_lamp", "cabinet_light", "accent_light"}
OUT_OF_VIEW = "out_of_view"
OFFSCREEN = "offscreen"
ROOMS = ["living_room", "bedroom", "kitchen", "bathroom", "hallway", "utility_room", "dressing_room", "other"]


def device_class(name):
    """Map a device id or name such as 'living_room_downlight' or 'accent_light_2' to a class."""
    n = (name or "").lower()
    for key, cls in [("fan", "exhaust_fan"), ("ceiling", "ceiling_light"), ("all_light", "ceiling_light"),
                     ("downlight", "downlight"), ("pendant", "pendant_light"), ("wall_light", "wall_light"),
                     ("sconce", "wall_light"), ("floor", "floor_lamp"), ("cabinet", "cabinet_light"),
                     ("accent", "accent_light"), ("indirect", "accent_light")]:
        if key in n:
            return cls
    return "other"


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

    def __init__(self, alpha=L["alpha"], floor=L["floor"]):
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
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        p = cls(d["alpha"])
        for lvl, table in d["counts"].items():
            for k, v in table.items():
                p.c[lvl][k] = Counter(v)
        return p


class HomeLinker:
    """Links for one room: keys [{id, gang_index, gang_count}], devices [{name, class}]."""

    def __init__(self, prior, room, keys, devices, out_of_view_share=L["out_of_view_share"], unused_p=L["unused_p"]):
        self.prior, self.room, self.keys = prior, room, keys
        self.unused_p = unused_p     # how likely a listed device is run by a switch elsewhere
        self.devices = devices
        self.ov = out_of_view_share
        self.fixed = {}          # key id -> device name, from a press
        self.ruled_out = defaultdict(set)

    def scores(self, key):
        """P(device | key) over this room's devices and 'out of view'."""
        p = self.prior.probs(self.room, key.get("gang_count"), key.get("gang_index"))
        listed = Counter(d["class"] for d in self.devices)
        n_accent = sum(v for c, v in listed.items() if c in ACCENT_GROUP)
        s = {}
        for d in self.devices:
            c = d["class"]
            v = p[c] / listed[c]
            if c in ACCENT_GROUP and c != "accent_light":
                v += p["accent_light"] / n_accent
            s[d["name"]] = v
        # classes with no listed device (a fan behind the ceiling, a light in another room)
        missing = sum(p[c] for c in CLASSES if listed[c] == 0 and not (c == "accent_light" and n_accent))
        s[OUT_OF_VIEW] = self.ov + missing
        for name in self.ruled_out[key["id"]]:
            s[name] = 1e-6
        tot = sum(s.values())
        return {k: v / tot for k, v in s.items()}

    def predict(self):
        free_keys = [k for k in self.keys if k["id"] not in self.fixed]
        taken = set(v for v in self.fixed.values() if v != OUT_OF_VIEW)
        names = [d["name"] for d in self.devices if d["name"] not in taken]
        sc = {k["id"]: self.scores(k) for k in free_keys}
        best = assign([k["id"] for k in free_keys], names, sc, self.unused_p)
        out = {}
        for k in self.keys:
            if k["id"] in self.fixed:
                out[k["id"]] = {"link": self.fixed[k["id"]], "p": 1.0, "verified": True, "ranked": []}
                continue
            ranked = sorted(sc[k["id"]].items(), key=lambda kv: -kv[1])
            link = best[k["id"]]
            out[k["id"]] = {"link": link, "p": round(sc[k["id"]][link], 3), "verified": False,
                            "ranked": [(n, round(v, 3)) for n, v in ranked[:4]]}
        return out

    def observe(self, key_id, changed_device, in_view=()):
        """Stage 3 feedback. changed_device is one of
        - a device name: that device changed, so the link is fixed;
        - OUT_OF_VIEW: nothing in the picture changed (a fan, or a light in another room);
        - OFFSCREEN: the picture got brighter or darker but no lamp in view changed, so the
          device is one of the listed lamps that are out of frame: rule out `in_view`;
        - None: unclear, nothing learned."""
        if changed_device is None:
            return False
        if changed_device == OFFSCREEN:
            self.ruled_out[key_id] |= set(in_view) | {OUT_OF_VIEW}
            return False
        before = self.predict()[key_id]["link"]
        self.fixed[key_id] = changed_device
        if changed_device == OUT_OF_VIEW:
            for d in self.devices:
                self.ruled_out[key_id].add(d["name"])
        return before != changed_device      # True when the guess had to be corrected


def assign(key_ids, names, sc, unused_p=L["unused_p"]):
    """Highest total log-probability, one device per key; out of view can repeat.
    Every listed device left without a key costs log(unused_p)."""
    m = len(names)
    memo = {}

    def go(i, used):
        if i == len(key_ids):
            return (m - bin(used).count("1")) * math.log(unused_p), ()
        if (i, used) in memo:
            return memo[(i, used)]
        s = sc[key_ids[i]]
        best = (-math.inf, ())
        opts = [(OUT_OF_VIEW, None)] + [(n, j) for j, n in enumerate(names) if not used >> j & 1]
        for n, j in opts:
            v, rest = go(i + 1, used if j is None else used | 1 << j)
            v += math.log(max(s.get(n, 1e-9), 1e-9))
            if v > best[0]:
                best = (v, (n,) + rest)
        memo[(i, used)] = best
        return best

    return dict(zip(key_ids, go(0, 0)[1]))


# ---------- team labels

def team_keys(dataset, split):
    for f in sorted(Path(dataset).glob("internal/*/labels.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("split") == "eval") != (split == "eval"):
                continue
            for c in r.get("controls") or []:
                role = c.get("role") or {}
                if c.get("kind") == "switch" and role.get("device_id"):
                    yield r.get("room_type") or "other", c.get("gang_count"), c.get("gang_index"), device_class(role["device_id"]), r


WEIGHT = L["label_weight"]


def fit(dataset):
    """One count per physical key (a burst shoots the same plate many times)."""
    p, seen = Prior(), set()
    for room, n, i, cls, r in team_keys(dataset, "revise"):
        k = (r.get("room_id"), n, i)
        if k in seen:
            continue
        seen.add(k)
        role = next(c["role"] for c in r["controls"] if c.get("gang_index") == i and c.get("role"))
        p.add(room, n, i, cls, WEIGHT.get(role.get("confidence"), 0.5))
    return p


def evaluate(dataset, prior):
    hit = rule_hit = n = 0
    rules_only = Prior()
    for room, gn, gi, cls, _ in team_keys(dataset, "eval"):
        n += 1
        p = prior.probs(room, gn, gi)
        q = rules_only.probs(room, gn, gi)
        ok = lambda guess: guess == cls or (cls == "accent_light" and guess in ACCENT_GROUP)
        hit += ok(max(p, key=p.get))
        rule_hit += ok(max(q, key=q.get))
    print(json.dumps({"team_eval_keys": n, "top1_agrees_with_team_label": round(hit / max(n, 1), 3),
                      "rules_only_top1": round(rule_hit / max(n, 1), 3)}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--dataset", default=str(path("dataset_root")))
    ap.add_argument("--save", default=str(path("prior")))
    a = ap.parse_args()
    if a.fit:
        fit(a.dataset).save(a.save)
        print("saved", a.save)
    if a.eval:
        evaluate(a.dataset, Prior.load(a.save) if Path(a.save).exists() else Prior())
