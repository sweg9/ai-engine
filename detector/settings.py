"""Read settings.yaml (C-12). Every number the detector uses comes from there."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path = ROOT / "settings.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["detector"]


def resolve(p: str) -> Path:
    """Paths in settings.yaml are relative to the repo root."""
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q).resolve()
