# ai-engine

## Interface for the hub

The hub in `backend-server` imports this package and calls it through one `ai_engine.Engine` object. The calls, their results and the settings they need are on the wiki: [ai-engine interface](https://sweg9.github.io/docs/api/ai-engine). It is not built yet.

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
