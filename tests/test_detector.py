"""Tests for the control detector pipeline (FR-01). They need no dataset and no weights."""

from detector.build_dataset import to_class, val_bursts, yolo_line
from detector.evaluate import iou, match

CFG = {
    "kind_map": {"switch": "switch", "toggle": "toggle", "push_button": "switch", "knob": None, "remote": None},
    "power_strip_sources": ["web-bing-powerstrip"],
}


def test_power_strip_switches_become_power_strip():
    assert to_class({"kind": "switch"}, "web-bing-powerstrip", CFG) == "power_strip"
    assert to_class({"kind": "switch"}, "ajm", CFG) == "switch"


def test_earlier_kinds_are_folded_or_dropped():
    assert to_class({"kind": "push_button"}, "ajm", CFG) == "switch"
    assert to_class({"kind": "toggle"}, "web-bing-powerstrip", CFG) == "toggle"
    assert to_class({"kind": "knob"}, "ajm", CFG) is None
    assert to_class({"kind": "remote"}, "ajm", CFG) is None


def test_yolo_line_uses_centre_and_clips_to_image():
    assert yolo_line(1, [0.1, 0.2, 0.2, 0.4]) == "1 0.200000 0.400000 0.200000 0.400000"
    assert yolo_line(0, [-0.1, 0.9, 0.3, 0.2]) == "0 0.100000 0.950000 0.200000 0.100000"


def test_val_keeps_whole_bursts_and_about_the_fraction():
    recs = [{"burst": f"b{i // 3}"} for i in range(300)]
    held = val_bursts(recs, 0.15, 0)
    n = sum(r["burst"] in held for r in recs)
    assert 45 <= n <= 48
    assert val_bursts(recs, 0.15, 0) == held          # same split on every run


def test_matching_class_aware_and_agnostic():
    gts = [(0, (0.1, 0.1, 0.2, 0.2)), (1, (0.5, 0.5, 0.7, 0.7))]
    preds = [(0, 0.9, (0.1, 0.1, 0.2, 0.21)), (0, 0.8, (0.5, 0.5, 0.7, 0.7))]
    assert iou(gts[0][1], preds[0][2]) > 0.9
    assert len(match(gts, preds, 0.5, class_aware=True)) == 1      # second has the wrong class
    assert len(match(gts, preds, 0.5, class_aware=False)) == 2     # but it found the control
