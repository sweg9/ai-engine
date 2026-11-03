# ai-engine
## Skills and standards

`skills/` is a git submodule of [cicd-harness](https://github.com/sweg9/cicd-harness). It holds the shared standards and skills for this project. Claude Code finds the skills through `.claude/skills`.

```bash
git clone --recurse-submodules <this repo>
git submodule update --init            # in a repo that is already cloned
git submodule update --remote skills   # move to the latest skills, then commit
```
