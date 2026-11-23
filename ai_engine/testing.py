"""`FakeEngine`: the same calls as `Engine`, answering from a script given in the test, so the
enclave program's tests need no model files.

    fake = FakeEngine()
    fake.script("check_effect", CheckResult(...))     # the next check_effect answers this
    fake.script("find_people", EngineError("down"))   # the next find_people raises this

A call with nothing scripted gives the safe not-trained answer; `find_people`, `detect_controls`
and `refind_control` raise `EngineError`, since an empty answer there could let a press through.
Every call is recorded in `fake.calls` as (name, args)."""

from collections import defaultdict, deque

from .results import (CheckResult, ClearResult, EngineError, PressPlan, RefindResult, StateResult)


class FakeEngine:
    def __init__(self):
        self._queue = defaultdict(deque)
        self.calls = []

    @classmethod
    def load(cls, settings=None, root=None):
        return cls()

    def script(self, call, *answers):
        if not hasattr(self, call) or call.startswith("_"):
            raise ValueError(f"FakeEngine has no call {call!r}")
        self._queue[call].extend(answers)
        return self

    def _answer(self, call, args, default):
        self.calls.append((call, args))
        if self._queue[call]:
            a = self._queue[call].popleft()
            if isinstance(a, Exception):
                raise a
            return a
        if isinstance(default, Exception):
            raise default
        return default

    def detect_controls(self, frame):
        return self._answer("detect_controls", (frame,), EngineError("nothing scripted for detect_controls"))

    def read_state(self, crop, kind):
        return self._answer("read_state", (crop, kind),
                            StateResult(state="unknown", confidence=0.0, reason="state classifier not trained"))

    def is_clear(self, crop, score):
        return self._answer("is_clear", (crop, score),
                            ClearResult(clear=False, sharpness=0.0, confidence=0.0, reason="nothing scripted"))

    def link_guess(self, controls, room):
        return self._answer("link_guess", (controls, room), [])

    def fingerprint(self, crop):
        return self._answer("fingerprint", (crop,), None)

    def measure_change(self, before, after):
        return self._answer("measure_change", (before, after), EngineError("nothing scripted for measure_change"))

    def detect_devices(self, image):
        return self._answer("detect_devices", (image,), [])

    def check_effect(self, before, after, device_box, device_type, action):
        return self._answer("check_effect", (before, after, device_box, device_type, action),
                            CheckResult(main="unsure", second=None, answer="unsure", disagree=False, measures={},
                                        confidence=0.0, reason="nothing scripted"))

    def find_people(self, image):
        return self._answer("find_people", (image,), EngineError("nothing scripted for find_people"))

    def refind_control(self, frame, saved):
        return self._answer("refind_control", (frame, saved),
                            RefindResult(found=False, box=None, similarity=None, confidence=0.0,
                                         reason="nothing scripted"))

    def plan_press(self, control, kind, actuators):
        return self._answer("plan_press", (control, kind, actuators),
                            PressPlan(actuator=None, push_ms=0, confidence=0.0, reason="nothing scripted"))

    def versions(self):
        return self._answer("versions", (), {"engine": "fake"})
