"""Run the link guess and the press-and-check loop on the synthetic scenes.

    python linker/sim.py [--checker learned] [--only syn-e41 syn-e42]

For each scene: guess every key's device (stage 2), then press the keys in the order they were
rendered and feed what change_check.py saw into HomeLinker.observe (stage 3). Reports how many
links are right before any press, after the presses, and how often a press corrected a guess.
The synthetic wiring is random, so the guess before any press is close to chance here; the
real prior can only be measured on real verified links.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "effect_check"))
from change_check import compare          # noqa: E402
from settings import path                 # noqa: E402
from linker import OFFSCREEN, OUT_OF_VIEW, HomeLinker, Prior, device_class   # noqa: E402


def scenes(root, only=None):
    for labels in sorted(Path(root).glob("syn-e*/labels.jsonl")):
        if only and labels.parent.name not in only:
            continue
        recs = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
        by_room = {}
        for r in recs:
            by_room.setdefault(r["room_id"], []).append(r)
        for rs in by_room.values():
            yield labels.parent, rs


def run(root, prior, compare, only=None):
    st = Counter()
    for folder, rs in scenes(root, only):
        first = rs[0]
        room = first["room_type"]
        devices = [{"name": d["name"], "class": device_class(d["name"])} for d in first["source"]["devices"]]
        names = {d["name"] for d in devices}
        keys, truth = [], {}
        for i, c in enumerate(first["controls"]):
            keys.append({"id": i, "gang_index": c["gang_index"], "gang_count": c["gang_count"]})
            dev = c["role"]["device_id"][len(room) + 1:]
            truth[i] = dev if dev in names else OUT_OF_VIEW
        L = HomeLinker(prior, room, keys, devices)
        guess = L.predict()
        st["keys"] += len(keys)
        st["right_before_press"] += sum(guess[k]["link"] == truth[k] for k in truth)
        st["chance"] += sum(1 / (len(devices) + 1) for _ in keys)
        # presses, in render order: the before record carries the evidence of the key pressed
        for r in rs:
            for i, c in enumerate(r["controls"]):
                ev = c.get("evidence")
                if not ev or ev["before_image"] != r["image_id"]:
                    continue
                img = folder / "images"
                vis = [{"name": d["name"], "bbox": d["bbox"]} for d in r["source"]["devices"]]
                res = compare(img / f"{ev['before_image']}.jpg", img / f"{ev['after_image']}.jpg", vis)
                in_view = [d["name"] for d in vis if d["bbox"]]
                seen = res["device"] or (OFFSCREEN if res["changed"] else OUT_OF_VIEW)
                st["presses"] += 1
                st["press_offscreen"] += seen == OFFSCREEN
                st["press_seen_right"] += seen == truth[i]
                st["corrected"] += bool(L.observe(i, seen, in_view))
        final = L.predict()
        st["right_after_press"] += sum(final[k]["link"] == truth[k] for k in truth)
        st["pressed_keys"] += sum(1 for k in truth if k in L.fixed)
    n = max(st["keys"], 1)
    print(json.dumps({
        "scenes_keys": st["keys"],
        "guess_right_before_press": round(st["right_before_press"] / n, 3),
        "chance": round(st["chance"] / n, 3),
        "presses": st["presses"],
        "press_offscreen": round(st["press_offscreen"] / max(st["presses"], 1), 3),
        "press_seen_right": round(st["press_seen_right"] / max(st["presses"], 1), 3),
        "guess_corrected_by_press": round(st["corrected"] / max(st["presses"], 1), 3),
        "right_after_press": round(st["right_after_press"] / n, 3),
    }, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default=str(path("synthetic_root")))
    ap.add_argument("--prior", default=str(path("prior")))
    ap.add_argument("--checker", choices=["simple", "learned"], default="simple")
    ap.add_argument("--only", nargs="*", help="seed folders to use, e.g. syn-e41 syn-e42")
    a = ap.parse_args()
    if a.checker == "learned":
        from learned import compare as learned_compare
        compare = learned_compare
    run(a.evidence, Prior.load(a.prior) if Path(a.prior).exists() else Prior(), compare, a.only)
