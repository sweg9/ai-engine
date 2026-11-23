"""Typed results of every `Engine` call (A-01, A-02). Frozen and checked when they are made:
a missing field or a value out of range raises `pydantic.ValidationError`."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

Kind = Literal["switch", "toggle", "power_strip"]
Answer = Literal["yes", "no", "unsure"]
Ratio = Annotated[float, Field(ge=0.0, le=1.0)]


def _box(v):
    x1, y1, x2, y2 = v
    if x2 < x1 or y2 < y1:
        raise ValueError(f"box corners out of order: {v}")
    return v


# [x1, y1, x2, y2] as ratios from 0 to 1 of the image size
Box = Annotated[tuple[Ratio, Ratio, Ratio, Ratio], AfterValidator(_box)]


class Result(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: Ratio
    reason: Annotated[str, Field(min_length=1)]


class EngineError(RuntimeError):
    """A model failed. The enclave answers the request with the reason; the hub fails the stage (A-04)."""


class ControlBox(Result):
    kind: Kind
    box: Box


class StateResult(Result):
    state: Literal["on", "off", "unknown"]


class ClearResult(Result):
    clear: bool
    sharpness: Annotated[float, Field(ge=0.0)]


class Gang(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    count: Annotated[int, Field(ge=1)]
    index: Annotated[int, Field(ge=0)]


class LinkGuess(Result):
    gang: Gang | None          # None for a control that is not a switch key
    device_class: str | None   # None when there is no guess (not a switch key)
    p: Ratio


class Fingerprint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vector: tuple[float, ...]
    model: Annotated[str, Field(min_length=1)]

    def similarity(self, other: "Fingerprint") -> float:
        if other.model != self.model:
            raise ValueError(f"fingerprints of different models: {self.model} and {other.model}")
        return float(sum(a * b for a, b in zip(self.vector, other.vector)))


class ChangeResult(Result):
    change_score: Ratio
    clear: bool
    direction: Literal["brighter", "darker", "none"]   # of the changed cells; none without a clear change


class DeviceBox(Result):
    type: Annotated[str, Field(min_length=1)]
    box: Box


class CheckResult(Result):
    main: Answer
    second: Answer | None
    answer: Answer
    disagree: bool
    measures: dict[str, float]


class PersonBox(Result):
    kind: Literal["person", "cat", "dog"]
    box: Box


class SavedControl(BaseModel):
    """What `refind_control` gets about the control it looks for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    box: Box
    fingerprint: Fingerprint | None = None


class RefindResult(Result):
    found: bool
    box: Box | None
    similarity: float | None   # None when no fingerprint could be compared


class Actuator(BaseModel):
    """One actuator of the hub settings and the part of the image it faces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    region: Box


class PressPlan(Result):
    actuator: str | None
    push_ms: Annotated[int, Field(ge=0)]
