"""Tests of ai_engine without model files (FR-01, FR-02, FR-08, FR-09, FR-10; A-01 to A-04)."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from ai_engine import (Actuator, CheckResult, ControlBox, Engine, EngineError, Fingerprint,
                       RefindResult, SavedControl)
from ai_engine.testing import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((ROOT / "settings.yaml").read_text())


def settings(**paths):
    s = copy.deepcopy(SETTINGS)
    s["paths"].update(detector_weights=None, people_weights=None)
    s["paths"].update(paths)
    return s


@pytest.fixture
def engine():
    return Engine.load(settings())


def img(value=128, h=60, w=80):
    return np.full((h, w, 3), value, dtype=np.uint8)


def noisy(h=60, w=80, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def switch(box, conf=0.9):
    return ControlBox(kind="switch", box=box, confidence=conf, reason="switch found with score 0.90")


# ---------- loading (C-12, A-04)

def test_load_fails_when_a_named_file_is_missing():
    with pytest.raises(EngineError, match="missing"):
        Engine.load(settings(detector_weights="models/no-such.pt"))


def test_load_fails_when_a_setting_is_missing():
    s = settings()
    del s["check"]["light"]["upper"]
    with pytest.raises(KeyError, match="check.light.upper"):
        Engine.load(s)


def test_load_needs_a_dict():
    with pytest.raises(ValueError):
        Engine.load("settings.yaml")


def test_versions_list_every_model(engine):
    v = engine.versions()
    assert v["detector"] == "not trained" and v["people"] == "not trained"
    assert v["prior"].startswith("prior.json sha256:")


# ---------- results are checked (A-01, A-02)

def test_results_reject_out_of_range_values():
    with pytest.raises(ValidationError):
        ControlBox(kind="switch", box=(0.1, 0.1, 1.2, 0.5), confidence=0.9, reason="x")
    with pytest.raises(ValidationError):
        ControlBox(kind="knob", box=(0.1, 0.1, 0.2, 0.5), confidence=0.9, reason="x")
    with pytest.raises(ValidationError):
        ControlBox(kind="switch", box=(0.1, 0.1, 0.2, 0.5), confidence=1.5, reason="x")
    with pytest.raises(ValidationError):
        ControlBox(kind="switch", box=(0.1, 0.1, 0.2, 0.5), confidence=0.5, reason="")


def test_results_are_frozen():
    c = switch((0.1, 0.1, 0.2, 0.2))
    with pytest.raises(ValidationError):
        c.kind = "toggle"


def test_fingerprints_of_different_models_are_not_compared():
    a = Fingerprint(vector=(1.0, 0.0), model="a")
    with pytest.raises(ValueError):
        a.similarity(Fingerprint(vector=(1.0, 0.0), model="b"))
    assert a.similarity(Fingerprint(vector=(0.6, 0.8), model="a")) == pytest.approx(0.6)


# ---------- bad input (A-04)

def test_bad_images_raise_value_error(engine):
    with pytest.raises(ValueError):
        engine.read_state(np.zeros((10, 10), dtype=np.uint8), "switch")
    with pytest.raises(ValueError):
        engine.read_state(np.zeros((0, 10, 3), dtype=np.uint8), "switch")
    with pytest.raises(ValueError):
        engine.read_state(img().astype(np.float32), "switch")


# ---------- not trained yet answers safely

def test_not_trained_answers(engine):
    assert engine.read_state(img(), "switch").state == "unknown"
    assert engine.detect_devices(img()) == []
    assert engine.fingerprint(img()) is None


def test_missing_detectors_fail_instead_of_answering_empty(engine):
    """FR-10, C-04: no people model must never read as nobody there."""
    with pytest.raises(EngineError, match="people"):
        engine.find_people(img())
    with pytest.raises(EngineError, match="control detector"):
        engine.detect_controls(img())


# ---------- S1 (FR-01)

def test_is_clear(engine):
    assert engine.is_clear(noisy(), 0.9).clear
    blurry = engine.is_clear(img(), 0.9)
    assert not blurry.clear and "blurry" in blurry.reason
    assert not engine.is_clear(noisy(), 0.2).clear
    with pytest.raises(ValueError):
        engine.is_clear(noisy(), 1.2)


# ---------- link guess (FR-02, M-S2a)

def test_link_guess_groups_keys_into_plates(engine):
    plate = [switch((0.40, 0.30, 0.45, 0.40)), switch((0.45, 0.30, 0.50, 0.40))]
    alone = switch((0.80, 0.30, 0.85, 0.40))
    strip = ControlBox(kind="power_strip", box=(0.1, 0.8, 0.3, 0.9), confidence=0.8, reason="x")
    g = engine.link_guess(plate + [alone, strip], "bathroom")
    assert [(x.gang.count, x.gang.index) for x in g[:3]] == [(2, 0), (2, 1), (1, 0)]
    assert g[0].device_class == "ceiling_light"
    assert g[1].device_class == "exhaust_fan"
    assert g[3].gang is None and g[3].device_class is None
    assert all(x.p < 1.0 for x in g)


def test_link_guess_needs_a_known_room(engine):
    with pytest.raises(ValueError):
        engine.link_guess([], "garage")


# ---------- S2 (FR-02, C-08)

def test_measure_change(engine):
    dark, lit = img(40), img(200)
    assert engine.measure_change(dark, lit).clear
    same = engine.measure_change(dark, dark)
    assert not same.clear and same.change_score == 0.0
    assert "brighter" in engine.measure_change(dark, lit).reason


# ---------- check (FR-08)

BOX = (0.25, 0.25, 0.75, 0.75)


def lamp(on):
    f = img(30)
    if on:
        f[15:45, 20:60] = 230
    return f


def test_light_on_is_yes(engine):
    r = engine.check_effect([lamp(False)], [lamp(True)], BOX, "light", "on")
    assert (r.main, r.second, r.answer, r.disagree) == ("yes", "yes", "yes", False)
    assert r.measures["brightness_change"] > 0.5


def test_light_that_stayed_off_is_no(engine):
    r = engine.check_effect([lamp(False)], [lamp(False)], BOX, "light", "on")
    assert r.answer == "no"


def test_light_off_uses_the_reversed_sign(engine):
    assert engine.check_effect([lamp(True)], [lamp(False)], BOX, "light", "off").answer == "yes"
    assert engine.check_effect([lamp(True)], [lamp(False)], BOX, "light", "on").answer == "no"


def test_unsure_main_falls_back_to_second(engine):
    before, after = img(30), img(30)
    after[15:45, 20:60] = 60           # the lamp area is a bit brighter: unsure
    after[:, :10] = 255                # the room got brighter elsewhere
    r = engine.check_effect([before], [after], BOX, "light", "on")
    assert r.main == "unsure" and r.answer == r.second


def test_main_and_second_disagree(engine):
    before, after = img(30), img(30)
    after[28:32, 38:42] = 255          # a tiny bright spot: lamp no, room no
    r = engine.check_effect([before], [after], (0.0, 0.0, 0.05, 0.05), "light", "on")
    assert r.main == "no"
    after2 = img(30)
    after2[:, :] = 90                  # whole room brighter, lamp corner too
    after2[0:3, 0:4] = 30              # but the lamp area itself stayed dark
    r2 = engine.check_effect([before], [after2], (0.0, 0.0, 0.05, 0.05), "light", "on")
    assert r2.main == "no" and r2.second == "yes" and r2.disagree and r2.answer == "no"


def test_fan_is_unsure_until_its_thresholds_are_set(engine):
    r = engine.check_effect([img(), img()], [noisy(seed=1), noisy(seed=2)], BOX, "fan", "on")
    assert r.answer == "unsure" and r.second is None and "not set" in r.reason


def test_fan_with_thresholds(engine):
    s = settings()
    s["check"]["fan"].update(upper=0.1, lower=0.01)
    e = Engine.load(s)
    assert e.check_effect([img(), img()], [noisy(seed=1), noisy(seed=2)], BOX, "fan", "on").answer == "yes"
    assert e.check_effect([img(), img()], [img(), img()], BOX, "fan", "on").answer == "no"


def test_other_device_types_are_unsure(engine):
    assert engine.check_effect([img()], [img(250)], BOX, "heater", "on").answer == "unsure"


def test_check_rejects_bad_input(engine):
    with pytest.raises(ValueError):
        engine.check_effect([], [img()], BOX, "light", "on")
    with pytest.raises(ValueError):
        engine.check_effect([img()], [img()], BOX, "light", "toggle")
    with pytest.raises(ValueError):
        engine.check_effect([img()], [img()], (0.5, 0.5, 0.2, 0.9), "light", "on")


# ---------- re-find and plan (FR-09)

class Detects(Engine):
    """An Engine whose detector answers a fixed list."""

    def __init__(self, found, **kw):
        base = Engine.load(settings(**kw))
        super().__init__(base.s, base.root, base._m, base._files)
        self.found = found

    def detect_controls(self, frame):
        return self.found


SAVED = SavedControl(kind="switch", box=(0.40, 0.30, 0.45, 0.40))


def test_refind_needs_a_fingerprint_by_default():
    e = Detects([switch((0.41, 0.31, 0.46, 0.41))])
    r = e.refind_control(img(), SAVED)
    assert not r.found and "no fingerprint" in r.reason


def test_refind_by_place_when_allowed():
    e = Detects([switch((0.41, 0.31, 0.46, 0.41))])
    e.s["refind"]["require_fingerprint"] = False
    r = e.refind_control(img(), SAVED)
    assert r.found and r.box == (0.41, 0.31, 0.46, 0.41)


def test_refind_fails_when_nothing_is_near():
    e = Detects([switch((0.80, 0.30, 0.85, 0.40)),
                 ControlBox(kind="toggle", box=(0.40, 0.30, 0.45, 0.40), confidence=0.9, reason="x")])
    e.s["refind"]["require_fingerprint"] = False
    assert not e.refind_control(img(), SAVED).found


class Fingerprints(Detects):
    """A Detects whose fingerprint call answers a fixed vector, as DINOv2 will."""

    def __init__(self, found, vector):
        super().__init__(found)
        self.vector = vector

    def fingerprint(self, crop):
        return Fingerprint(vector=self.vector, model="test")


def saved_with(vector):
    return SavedControl(kind="switch", box=(0.40, 0.30, 0.45, 0.40),
                        fingerprint=Fingerprint(vector=vector, model="test"))


def test_refind_below_the_similarity_cancels():
    """FR-09: similarity under fingerprint.similarity_to_refind (0.7) means not found."""
    e = Fingerprints([switch((0.41, 0.31, 0.46, 0.41))], (0.6, 0.8))
    r = e.refind_control(img(), saved_with((1.0, 0.0)))
    assert not r.found and r.box is None and r.similarity == pytest.approx(0.6)
    assert "looks different" in r.reason


def test_refind_at_or_above_the_similarity_finds():
    e = Fingerprints([switch((0.41, 0.31, 0.46, 0.41))], (0.8, 0.6))
    r = e.refind_control(img(), saved_with((1.0, 0.0)))
    assert r.found and r.similarity == pytest.approx(0.8) and r.box == (0.41, 0.31, 0.46, 0.41)


def test_refind_threshold_comes_from_settings():
    e = Fingerprints([switch((0.41, 0.31, 0.46, 0.41))], (0.8, 0.6))
    e.s["fingerprint"]["similarity_to_refind"] = 0.9
    assert not e.refind_control(img(), saved_with((1.0, 0.0))).found


ACTUATORS = [Actuator(id="A1", region=(0.0, 0.0, 0.5, 1.0)), Actuator(id="A2", region=(0.5, 0.0, 1.0, 1.0))]


def found(box):
    return RefindResult(found=True, box=box, similarity=0.9, confidence=0.9, reason="x")


def test_plan_press_picks_the_facing_actuator(engine):
    p = engine.plan_press(found((0.6, 0.3, 0.7, 0.4)), "switch", ACTUATORS)
    assert p.actuator == "A2" and p.push_ms == 400


def test_plan_press_cancels_without_a_facing_actuator(engine):
    assert engine.plan_press(found((0.6, 0.3, 0.7, 0.4)), "switch", ACTUATORS[:1]).actuator is None
    lost = RefindResult(found=False, box=None, similarity=None, confidence=0.9, reason="x")
    assert engine.plan_press(lost, "switch", ACTUATORS).actuator is None


# ---------- FakeEngine

def test_fake_engine_follows_the_script():
    fake = FakeEngine()
    yes = CheckResult(main="yes", second=None, answer="yes", disagree=False, measures={}, confidence=0.9, reason="x")
    fake.script("check_effect", yes)
    assert fake.check_effect([img()], [img()], BOX, "light", "on") is yes
    assert fake.check_effect([img()], [img()], BOX, "light", "on").answer == "unsure"
    with pytest.raises(EngineError):
        fake.find_people(img())
    fake.script("find_people", [])
    assert fake.find_people(img()) == []
    assert [c for c, _ in fake.calls] == ["check_effect", "check_effect", "find_people", "find_people"]


def test_fake_engine_has_every_call():
    calls = {n for n in dir(Engine) if not n.startswith("_") and n != "load"}
    assert calls <= set(dir(FakeEngine))


# ---------- settings (C-12, C-13)

def test_unknown_planner_fails_the_load():
    s = settings()
    s["press"]["planner"] = "vla"
    with pytest.raises(EngineError, match="press.planner"):
        Engine.load(s)


def test_change_grid_comes_from_settings():
    s = settings()
    s["change"]["grid"] = 2
    before, after = img(40), img(40)
    after[:30, :40] = 200              # one quarter of the region
    assert Engine.load(s).measure_change(before, after).change_score == pytest.approx(0.25)
