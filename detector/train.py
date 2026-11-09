"""Train the control detector (FR-01) on the YOLO dataset from build_dataset.py."""
import argparse
from pathlib import Path

from ultralytics import YOLO

from detector import settings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", type=Path)
    ap.add_argument("--name", default="detector")
    ap.add_argument("--epochs", type=int, help="override settings, e.g. for a smoke run")
    args = ap.parse_args()
    cfg = settings.load(args.settings)["detector"]
    t = dict(cfg["train"])
    if args.epochs:
        t["epochs"] = args.epochs
    data = settings.resolve(cfg["build_dir"]) / "data.yaml"
    model = YOLO(t.pop("model"))
    model.train(
        data=str(data),
        project=str(settings.REPO / "runs"),
        name=args.name,
        exist_ok=False,
        cache="ram",       # disk 캐시는 데이터셋 레포에 .npy를 만들므로 쓰지 않는다
        fliplr=0.5,
        degrees=5,
        **t,
    )


if __name__ == "__main__":
    main()
