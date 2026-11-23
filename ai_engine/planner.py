"""Motion planners behind `Engine.plan_press` (FR-09). Swappable by `press.planner` in the settings
(C-13): a simple script now, a VLA model later behind the same `plan` call."""

from .results import PressPlan


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class ScriptPlanner:
    """The actuator whose part of the image holds the control's center; the push time of its kind."""

    def __init__(self, settings):
        self.push_ms = settings["press"]["push_ms"]

    def plan(self, control, kind, actuators):
        push_ms = int(self.push_ms[kind])
        if not control.found or control.box is None:
            return PressPlan(actuator=None, push_ms=push_ms, confidence=1.0, reason=f"the {kind} was not found again")
        cx, cy = _center(control.box)
        facing = [a for a in actuators if a.region[0] <= cx <= a.region[2] and a.region[1] <= cy <= a.region[3]]
        if not facing:
            return PressPlan(actuator=None, push_ms=push_ms, confidence=1.0, reason=f"no actuator faces the {kind}")
        best = min(facing, key=lambda a: sum((p - q) ** 2 for p, q in zip(_center(a.region), (cx, cy))))
        return PressPlan(actuator=best.id, push_ms=push_ms, confidence=control.confidence,
                         reason=f"actuator {best.id} faces the {kind}")


PLANNERS = {"script": ScriptPlanner}
