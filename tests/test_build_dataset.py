"""FR-01: the detector dataset keeps the five kinds and the revise/eval split."""
from detector.build_dataset import assign, yolo_lines
from detector.evaluate import match

CLASSES = ["switch", "push_button", "toggle", "knob", "remote"]


def rec(split="revise", burst="b1", controls=()):
    return {"image_id": "x-0001", "split": split, "burst": burst, "controls": list(controls)}


def test_eval_split_goes_only_to_test():
    assert assign(rec(split="eval"), val_fraction=1.0) == "test"


def test_revise_never_goes_to_test():
    assert assign(rec(), 0.0) == "train"
    assert assign(rec(), 1.0) == "val"


def test_same_burst_same_side():
    a = assign(rec(burst="ajm-b7"), 0.5)
    assert all(assign(rec(burst="ajm-b7"), 0.5) == a for _ in range(5))


def test_box_converted_and_clipped():
    c = {"kind": "toggle", "bbox": [0.9, 0.1, 0.2, 0.2], "decoy": False}
    (line,) = yolo_lines(rec(controls=[c]), CLASSES)
    k, cx, cy, w, h = line.split()
    assert k == "2"
    assert abs(float(w) - 0.1) < 1e-6 and abs(float(cx) - 0.95) < 1e-6


def test_decoy_not_labelled():
    c = {"kind": "switch", "bbox": [0.1, 0.1, 0.2, 0.2], "decoy": True}
    assert yolo_lines(rec(controls=[c]), CLASSES) == []


def test_match_counts_wrong_kind_as_miss():
    gts = [("switch", (0, 0, 0.5, 0.5))]
    preds = [("toggle", 0.9, (0, 0, 0.5, 0.5))]
    m = match(gts, preds, 0.5)
    assert m["toggle"] == [0, 1, 0] and m["switch"] == [0, 0, 1]
