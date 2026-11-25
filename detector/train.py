"""Train the control detector (FR-01) with the values in settings.yaml.

    python detector/build_yolo.py      # renders the demo bedroom set first if it is missing (Blender)
    python detector/train.py           # writes runs/controls/weights/best.pt (git-ignored)

Copy the chosen weights to the `detector_weights` path in settings.yaml.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import S, path   # noqa: E402
from ultralytics import YOLO   # noqa: E402

if __name__ == "__main__":
    t = S["detector"]["train"]
    YOLO(t["model"]).train(data=str(path("build_dir") / "data.yaml"), imgsz=t["imgsz"], epochs=t["epochs"],
                           batch=t["batch"], device=t["device"], workers=t["workers"], patience=t["patience"],
                           project=str(ROOT / "runs"), name="controls", exist_ok=True)
