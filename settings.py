"""Read settings.yaml (C-12). `S` is the whole file; `path` resolves a path setting."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
S = yaml.safe_load((ROOT / "settings.yaml").read_text())


def path(name):
    return (ROOT / S["paths"][name]).resolve()
