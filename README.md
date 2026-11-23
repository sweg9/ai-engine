# ai-engine

## Interface for the enclave

The enclave program in `backend-server` (`src/backend/enclave/`) imports this package and calls it through one `ai_engine.Engine` object. The hub and the server never import it. The enclave gets its settings over vsock at start, so no image holds them. The calls, their results and the settings they need are on the wiki: [ai-engine interface](https://sweg9.github.io/docs/api/ai-engine).

```python
import yaml
from ai_engine import Engine

engine = Engine.load(yaml.safe_load(settings_text))   # the enclave gets settings_text over vsock
controls = engine.detect_controls(frame)              # frame: uint8 BGR array
guesses = engine.link_guess(controls, "living_room")
```

| Call | Now |
|---|---|
| `detect_controls`, `link_guess`, `is_clear`, `measure_change`, `find_people`, `plan_press` | Work. `detect_controls` and `find_people` need their weights in `models/` (git-ignored) and raise `EngineError` without them, so a missing model never reads as "nothing there" |
| `check_effect` | The brightness measure for `light` (lamp area, then the whole frame as the second check). `fan` answers `unsure` until `check.fan` thresholds are measured. The learned MobileNet check (`effect_check/learned.py`) is not wired in yet |
| `read_state`, `detect_devices` | Not trained: `unknown` and an empty list |
| `fingerprint` | Not built (DINOv2): `None`. So `refind_control` finds nothing while `refind.require_fingerprint` is `true` (FR-09) |

`ai_engine.testing.FakeEngine` answers from a script for the enclave program's tests. The package installs alone with `pip install .` (add `[models]` for YOLO); `python -m pytest` runs its tests without model files.

## Skills and standards

`skills/` is a git submodule of [harness](https://github.com/sweg9/harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```

## Branch yolo11n-640-synth

Control detector (YOLO11n), link guess and before/after effect check, with a camera demo (FR-01, FR-02).
Model card and scores: [reports/yolo11n-640-synth.md](reports/yolo11n-640-synth.md).

```bash
pip install -r requirements.txt                 # Python 3.11
python detector/build_yolo.py                   # needs ../dataset and ../synthetic/evidence
python detector/train.py
python linker/linker.py --fit --eval
python effect_check/learned.py --train --eval
python linker/sim.py --checker learned --only syn-e41 syn-e42
python demo/server.py                           # http://localhost:8000, camera in Chrome
```

Settings are in `settings.yaml` (C-12). Weights go in `models/` and are git-ignored.
Synthetic renders: `synthetic/run_par.sh` with Blender 5.2.
