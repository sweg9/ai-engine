"""`Engine`: every model step the enclave calls (wiki: API > ai-engine interface).

The enclave program gets the settings over vsock at start and calls `Engine.load(settings)` once.
Every number comes from the settings (C-12). A model whose weights are `null` is not trained
yet and its call still answers safely; a model file that is named but missing fails `load`."""

import hashlib
from pathlib import Path

import numpy as np

from . import images
from .link import ROOMS, Prior, plates
from .results import (Actuator, ChangeResult, CheckResult, ClearResult, ControlBox, DeviceBox,
                      EngineError, Fingerprint, Gang, LinkGuess, PersonBox, PressPlan,
                      RefindResult, SavedControl, StateResult)

REPO_ROOT = Path(__file__).resolve().parents[1]
KINDS = ("switch", "toggle", "power_strip")
PEOPLE = {"person", "cat", "dog"}


def _get(settings, dotted, default=KeyError):
    node = settings
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is KeyError:
                raise KeyError(f"setting {dotted} is missing")
            return default
        node = node[part]
    return node


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _clip01(v):
    return float(min(1.0, max(0.0, v)))


def _decide(value, upper, lower):
    """yes at or above `upper`, no at or below `lower`, unsure between (A-03)."""
    span = max(upper - lower, 1e-6)
    if value >= upper:
        return "yes", _clip01(0.75 + 0.25 * (value - upper) / span)
    if value <= lower:
        return "no", _clip01(0.75 + 0.25 * (lower - value) / span)
    return "unsure", _clip01(0.5 * min(value - lower, upper - value) / span + 0.25)


class Engine:
    def __init__(self, settings, root, models, files):
        self.s = settings
        self.root = root
        self._m = models        # name -> loaded model or None
        self._files = files     # name -> Path of the loaded file

    # ---------- loading

    @classmethod
    def load(cls, settings, root=None):
        """Load every model file named in `settings` (a dict, as parsed from `settings.yaml`).
        Paths are relative to `root`, the repo root by default. Fails at once if a named file
        is missing or a needed setting is absent."""
        if not isinstance(settings, dict):
            raise ValueError("settings must be a dict parsed from ai-engine's settings.yaml")
        root = Path(root) if root else REPO_ROOT
        files, models = {}, {}
        device = _get(settings, "engine.device", "cpu")

        def file(name, dotted):
            rel = _get(settings, dotted)
            if rel is None:
                return None
            p = (root / rel).resolve()
            if not p.exists():
                raise EngineError(f"{name} file {rel} named in {dotted} is missing")
            files[name] = p
            return p

        def yolo(p):
            if p is None:
                return None
            try:
                from ultralytics import YOLO
                m = YOLO(str(p))
                m.to(device)
                return m
            except Exception as e:   # a broken file fails the load, not a run (A-04)
                raise EngineError(f"could not load {p.name}: {e}") from e

        models["detector"] = yolo(file("detector", "paths.detector_weights"))
        models["state"] = yolo(file("state", "paths.state_weights"))
        models["devices"] = yolo(file("devices", "paths.devices_weights"))
        models["people"] = yolo(file("people", "paths.people_weights"))
        if _get(settings, "paths.fingerprint_weights") is not None:
            raise EngineError("paths.fingerprint_weights is set, but DINOv2 fingerprints are not built yet; set it to null")
        models["fingerprint"] = None
        prior = file("prior", "paths.prior")
        floor = _get(settings, "linker.floor")
        alpha = _get(settings, "linker.alpha")
        models["prior"] = Prior.load(prior, floor) if prior else Prior(alpha, floor)
        for key in ("detector.predict.conf", "detector.predict.imgsz", "linker.plate_gap",
                    "state.threshold", "clear.min_score", "clear.min_sharpness",
                    "fingerprint.similarity_to_refind", "change.threshold", "change.cell_min",
                    "devices.conf", "check.light.upper", "check.light.lower",
                    "check.light.scene_upper", "check.light.scene_lower", "check.fan.upper",
                    "check.fan.lower", "people.conf", "refind.max_shift",
                    "refind.require_fingerprint", "press.push_ms"):
            _get(settings, key)
        return cls(settings, root, models, files)

    def _yolo(self, name, img, conf):
        m = self._m[name]
        try:
            r = m.predict(img, imgsz=self.s["detector"]["predict"]["imgsz"], conf=conf, verbose=False)[0]
        except Exception as e:
            raise EngineError(f"{name} model failed: {e}") from e
        return r

    @staticmethod
    def _boxes(r, img):
        h, w = img.shape[:2]
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            box = (_clip01(x1 / w), _clip01(y1 / h), _clip01(x2 / w), _clip01(y2 / h))
            yield r.names[int(c)], box, _clip01(s)

    # ---------- S1

    def detect_controls(self, frame):
        """Controls in one photo (FR-01, M-S1a)."""
        images.check_image(frame, "frame")
        if self._m["detector"] is None:
            raise EngineError("the control detector is not loaded (paths.detector_weights is null)")
        d = self.s["detector"]
        kind_map = d.get("kind_map")
        out = []
        for name, box, score in self._boxes(self._yolo("detector", frame, d["predict"]["conf"]), frame):
            kind = kind_map.get(name) if kind_map else name
            if kind not in KINDS:
                continue
            out.append(ControlBox(kind=kind, box=box, confidence=score,
                                  reason=f"{kind} found with score {score:.2f}"))
        return out

    def read_state(self, crop, kind):
        """On, off or unknown for one control (FR-01, M-S1b)."""
        images.check_image(crop, "crop")
        if kind not in KINDS:
            raise ValueError(f"unknown control kind {kind!r}")
        if self._m["state"] is None:
            return StateResult(state="unknown", confidence=0.0, reason="state classifier not trained")
        try:
            r = self._m["state"].predict(crop, verbose=False)[0]
            name, conf = r.names[int(r.probs.top1)], float(r.probs.top1conf)
        except Exception as e:
            raise EngineError(f"state classifier failed: {e}") from e
        if name not in ("on", "off"):
            raise EngineError(f"state classifier answered an unknown class {name!r}")
        if conf < self.s["state"]["threshold"]:
            return StateResult(state="unknown", confidence=_clip01(conf), reason=f"not sure if {name} ({conf:.2f})")
        return StateResult(state=name, confidence=_clip01(conf), reason=f"looks {name} ({conf:.2f})")

    def is_clear(self, crop, score):
        """Clear enough to save (FR-01): the detector's score and the sharpness both pass."""
        images.check_image(crop, "crop")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be from 0 to 1, got {score}")
        c = self.s["clear"]
        sharp = images.sharpness(crop)
        if score < c["min_score"]:
            return ClearResult(clear=False, sharpness=sharp, confidence=_clip01(1 - score),
                               reason=f"score too low ({score:.2f} < {c['min_score']})")
        if sharp < c["min_sharpness"]:
            return ClearResult(clear=False, sharpness=sharp, confidence=_clip01(1 - sharp / c["min_sharpness"] / 2),
                               reason=f"too blurry (sharpness {sharp:.0f} < {c['min_sharpness']})")
        return ClearResult(clear=True, sharpness=sharp, confidence=_clip01(score),
                           reason=f"clear (score {score:.2f}, sharpness {sharp:.0f})")

    def link_guess(self, controls, room):
        """Which device class each switch key most likely runs, before any press (FR-02, M-S2a).
        One answer per control, in the same order."""
        if room not in ROOMS:
            raise ValueError(f"unknown room type {room!r}; one of {ROOMS}")
        gangs = plates([(c.kind, c.box) for c in controls], self.s["linker"]["plate_gap"])
        out = []
        for c, g in zip(controls, gangs):
            if g is None:
                out.append(LinkGuess(gang=None, device_class=None, p=0.0, confidence=0.0,
                                     reason=f"no link guess for a {c.kind}"))
                continue
            count, index = g
            probs = self._m["prior"].probs(room, count, index)
            cls, p = max(probs.items(), key=lambda kv: kv[1])
            out.append(LinkGuess(gang=Gang(count=count, index=index), device_class=cls, p=_clip01(p),
                                 confidence=_clip01(p),
                                 reason=f"key {index + 1} of {count} in a {room.replace('_', ' ')}: "
                                        f"{cls.replace('_', ' ')} ({p:.2f})"))
        return out

    def fingerprint(self, crop):
        """A vector to tell one control from another (FR-01, FR-09). None until DINOv2 is built:
        the hub then stores no fingerprint."""
        images.check_image(crop, "crop")
        return None

    # ---------- S2

    def measure_change(self, before, after):
        """How much a region changed after the resident used a control (FR-02, C-08)."""
        images.check_image(before, "before")
        images.check_image(after, "after")
        c = self.s["change"]
        grid = 8
        size = (max(grid, before.shape[0] // grid * grid), max(grid, before.shape[1] // grid * grid))
        b = images.resize(images.gray(before), size)
        a = images.resize(images.gray(after), size)
        d = a - b
        cells = d.reshape(grid, size[0] // grid, grid, size[1] // grid).mean(axis=(1, 3))
        changed = np.abs(cells) >= c["cell_min"]
        score = float(changed.mean())
        clear = score >= c["threshold"]
        mean = float(cells[changed].mean()) if changed.any() else 0.0
        how = "brighter" if mean > 0 else "darker"
        reason = (f"{score:.0%} of the region got {how}" if changed.any() else "the region did not change")
        conf = _clip01(0.5 + abs(score - c["threshold"]) / max(c["threshold"], 1e-6) / 2)
        return ChangeResult(change_score=_clip01(score), clear=clear, confidence=conf, reason=reason)

    def detect_devices(self, image):
        """Devices in the region that changed (FR-02, M-S2). Empty until the model exists."""
        images.check_image(image, "image")
        if self._m["devices"] is None:
            return []
        return [DeviceBox(type=name, box=box, confidence=score, reason=f"{name} found with score {score:.2f}")
                for name, box, score in self._boxes(self._yolo("devices", image, self.s["devices"]["conf"]), image)]

    # ---------- S5, S6

    def check_effect(self, before, after, device_box, device_type, action):
        """Did the command work, seen by the camera (FR-08, M-S5 baseline measure)."""
        for name, frames in (("before", before), ("after", after)):
            if not frames:
                raise ValueError(f"{name} needs at least one frame")
            for f in frames:
                images.check_image(f, name)
        images.check_box(device_box, "device_box")
        if action not in ("on", "off"):
            raise ValueError(f"action must be on or off, got {action!r}")
        sign = 1.0 if action == "on" else -1.0
        measures, main, second, why = {}, "unsure", None, []
        main_conf, second_conf = 0.0, 0.0

        if device_type == "light":
            t = self.s["check"]["light"]
            dev = sign * self._brightness([images.crop(f, device_box) for f in before],
                                          [images.crop(f, device_box) for f in after])
            room = sign * self._brightness(before, after)
            measures.update(brightness_change=dev, scene_brightness_change=room)
            main, main_conf = _decide(dev, t["upper"], t["lower"])
            second, second_conf = _decide(room, t["scene_upper"], t["scene_lower"])
            word = "brighter" if action == "on" else "darker"
            why = [f"lamp area {dev:.0%} {word}", f"room {room:.0%} {word}"]
        elif device_type == "fan":
            t = self.s["check"]["fan"]
            motion = sign * (self._motion([images.crop(f, device_box) for f in after])
                             - self._motion([images.crop(f, device_box) for f in before]))
            measures.update(motion_change=motion)
            if len(before) < 2 or len(after) < 2:
                why = ["a fan needs at least two frames before and after"]
            elif t["upper"] is None or t["lower"] is None:
                why = ["fan thresholds are not set yet"]
            else:
                main, main_conf = _decide(motion, t["upper"], t["lower"])
                why = [f"fan area motion changed by {motion:.3f}"]
        else:
            why = [f"no camera check for a {device_type}"]

        answer, conf = (main, main_conf) if main != "unsure" or second is None else (second, second_conf)
        disagree = main in ("yes", "no") and second in ("yes", "no") and main != second
        return CheckResult(main=main, second=second, answer=answer, disagree=disagree, measures=measures,
                           confidence=_clip01(conf), reason="; ".join(why))

    @staticmethod
    def _brightness(before, after):
        b = float(np.mean([images.gray(f).mean() for f in before]))
        a = float(np.mean([images.gray(f).mean() for f in after]))
        return a - b

    @staticmethod
    def _motion(frames):
        if len(frames) < 2:
            return 0.0
        g = [images.gray(f) for f in frames]
        return float(np.mean([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)]))

    def find_people(self, image):
        """People and pets in a photo (FR-10, C-04, C-05). Without the model this fails rather
        than answering that nobody is there."""
        images.check_image(image, "image")
        if self._m["people"] is None:
            raise EngineError("the people detector is not loaded (paths.people_weights is null)")
        return [PersonBox(kind=name, box=box, confidence=score, reason=f"{name} found with score {score:.2f}")
                for name, box, score in self._boxes(self._yolo("people", image, self.s["people"]["conf"]), image)
                if name in PEOPLE]

    def refind_control(self, frame, saved):
        """Find the saved control again at the press spot (FR-09)."""
        if not isinstance(saved, SavedControl):
            saved = SavedControl.model_validate(saved)
        r = self.s["refind"]
        near = [c for c in self.detect_controls(frame)
                if c.kind == saved.kind
                and max(abs(a - b) for a, b in zip(_center(c.box), _center(saved.box))) <= r["max_shift"]]
        if not near:
            return RefindResult(found=False, box=None, similarity=None, confidence=0.9,
                                reason=f"no {saved.kind} near the saved place")
        best = max(near, key=lambda c: _iou(c.box, saved.box))
        mine = self.fingerprint(images.crop(frame, best.box))
        if mine is not None and saved.fingerprint is not None:
            sim = mine.similarity(saved.fingerprint)
            need = self.s["fingerprint"]["similarity_to_refind"]
            if sim < need:
                return RefindResult(found=False, box=None, similarity=sim, confidence=_clip01(1 - sim),
                                    reason=f"the {saved.kind} near the saved place looks different ({sim:.2f} < {need})")
            return RefindResult(found=True, box=best.box, similarity=sim, confidence=_clip01(min(sim, best.confidence)),
                                reason=f"{saved.kind} found again (similarity {sim:.2f})")
        if r["require_fingerprint"]:
            return RefindResult(found=False, box=None, similarity=None, confidence=0.5,
                                reason=f"a {saved.kind} is near the saved place, but no fingerprint to compare")
        return RefindResult(found=True, box=best.box, similarity=None, confidence=_clip01(best.confidence * 0.8),
                            reason=f"{saved.kind} found again by place only (no fingerprint)")

    def plan_press(self, control, kind, actuators):
        """Which actuator presses and how long (FR-09). A script now, swappable later (C-13)."""
        if kind not in KINDS:
            raise ValueError(f"unknown control kind {kind!r}")
        acts = [a if isinstance(a, Actuator) else Actuator.model_validate(a) for a in actuators]
        push_ms = int(self.s["press"]["push_ms"][kind])
        if not control.found or control.box is None:
            return PressPlan(actuator=None, push_ms=push_ms, confidence=1.0, reason=f"the {kind} was not found again")
        cx, cy = _center(control.box)
        facing = [a for a in acts if a.region[0] <= cx <= a.region[2] and a.region[1] <= cy <= a.region[3]]
        if not facing:
            return PressPlan(actuator=None, push_ms=push_ms, confidence=1.0, reason=f"no actuator faces the {kind}")
        best = min(facing, key=lambda a: sum((p - q) ** 2 for p, q in zip(_center(a.region), (cx, cy))))
        return PressPlan(actuator=best.id, push_ms=push_ms, confidence=control.confidence,
                         reason=f"actuator {best.id} faces the {kind}")

    # ---------- reports

    def versions(self):
        """Model file names and hashes, for reports and model cards (A-05)."""
        out = {}
        for name, p in sorted(self._files.items()):
            out[name] = f"{p.name} sha256:{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}"
        for name in ("detector", "state", "devices", "people", "fingerprint"):
            out.setdefault(name, "not trained")
        return out
