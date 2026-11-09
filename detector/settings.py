"""settings.yaml 로더 (C-12)."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def load(path: Path | None = None) -> dict:
    with open(path or REPO / "settings.yaml") as f:
        return yaml.safe_load(f)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (REPO / q).resolve()
