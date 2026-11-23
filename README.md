# ai-engine

## Interface for the enclave

The enclave program in `backend-server` (`src/backend/enclave/`) imports this package and calls it through one `ai_engine.Engine` object. The hub and the server never import it. The enclave gets its settings over vsock at start, so no image holds them. The calls, their results and the settings they need are on the wiki: [ai-engine interface](https://sweg9.github.io/docs/api/ai-engine). It is not built yet.

## Skills and standards

`skills/` is a git submodule of [harness](https://github.com/sweg9/harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```
