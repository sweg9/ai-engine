"""Run the detector on a camera or a video file (FR-01).

Usage:
    python -m detector.infer runs/detector-yolo11n/weights/best.pt --source 0          # webcam 0
    python -m detector.infer runs/detector-yolo11n/weights/best.pt --source clip.mp4 --show

Every `frame_stride`-th frame is checked (FR-01: every 5th frame, to save computing power).
For each checked frame one JSON line is printed: the frame number and the controls found,
each with its kind (`switch`, `toggle` or `power_strip`), a box in ratios of the frame and a
confidence. Frames stay on this machine; nothing is sent or saved (C-05).
"""

import argparse
import json

import cv2
from ultralytics import YOLO

from detector.settings import load


def detections(model, frame, cfg) -> list:
    res = model.predict(frame, conf=cfg["predict"]["conf"], imgsz=cfg["train"]["imgsz"],
                        device=cfg["train"]["device"], verbose=False)[0]
    return [{"kind": cfg["classes"][int(c)], "bbox": [round(v, 4) for v in b], "confidence": round(s, 3)}
            for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxyn.tolist())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--show", action="store_true", help="draw the boxes in a window")
    args = ap.parse_args()
    cfg = load()
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    n, last = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % cfg["frame_stride"] == 0:
            last = detections(model, frame, cfg)
            print(json.dumps({"frame": n, "controls": last}), flush=True)
        if args.show:
            h, w = frame.shape[:2]
            for d in last:
                x1, y1, x2, y2 = d["bbox"]
                p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
                cv2.rectangle(frame, p1, p2, (0, 200, 255), 2)
                cv2.putText(frame, f"{d['kind']} {d['confidence']:.2f}", (p1[0], max(12, p1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.imshow("controls", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        n += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
