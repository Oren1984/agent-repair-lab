"""Strict patch schema, validation and application.

A proposal is a JSON object::

    {
      "diagnosis": "short evidence-based explanation",
      "edits": [
        {"parameter": "W2", "index": [row, col],
         "expected_original": <float>, "new_value": <float>}
      ]
    }

Edits carry absolute ``new_value``s. The whole proposal is validated before
any value is written, and values are only ever written to a copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .policy import ACTIONS, HIDDEN_UNITS, MLPPolicy


@dataclass(frozen=True)
class PatchConstraints:
    allowed_parameters: tuple[str, ...] = ("W2",)
    max_edits: int = 3
    max_abs_delta: float = 3.0
    original_tolerance: float = 1e-3  # baseline weights are stored with 3 decimals
    max_diagnosis_chars: int = 2000

    def to_dict(self) -> dict:
        return {
            "allowed_parameters": list(self.allowed_parameters),
            "max_edits": self.max_edits,
            "max_abs_delta": self.max_abs_delta,
            "original_tolerance": self.original_tolerance,
            "value_representation": "absolute new_value (delta = new_value - expected_original)",
        }


@dataclass(frozen=True)
class Edit:
    parameter: str
    index: tuple[int, ...]
    expected_original: float
    new_value: float

    @property
    def delta(self) -> float:
        return self.new_value - self.expected_original

    def label(self) -> str:
        if self.parameter == "W2" and len(self.index) == 2:
            return f"W2[{ACTIONS[self.index[0]]} <- {HIDDEN_UNITS[self.index[1]]}]"
        return f"{self.parameter}{list(self.index)}"


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)
    diagnosis: str = ""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_proposal(raw, policy: MLPPolicy, constraints: PatchConstraints = PatchConstraints()) -> ValidationResult:
    errors: list[str] = []
    if not isinstance(raw, dict):
        return ValidationResult(False, ["proposal must be a JSON object"])

    diagnosis = raw.get("diagnosis")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        errors.append("diagnosis must be a non-empty string")
        diagnosis = ""
    elif len(diagnosis) > constraints.max_diagnosis_chars:
        errors.append(f"diagnosis exceeds {constraints.max_diagnosis_chars} characters")

    raw_edits = raw.get("edits")
    if not isinstance(raw_edits, list):
        errors.append("edits must be a list")
        return ValidationResult(False, errors, diagnosis=diagnosis)
    if not raw_edits:
        errors.append("proposal contains no edits")
    if len(raw_edits) > constraints.max_edits:
        errors.append(f"{len(raw_edits)} edits exceed the limit of {constraints.max_edits}")

    edits: list[Edit] = []
    seen: set[tuple] = set()
    for i, item in enumerate(raw_edits):
        where = f"edit {i}"
        if not isinstance(item, dict):
            errors.append(f"{where}: must be an object")
            continue
        missing = {"parameter", "index", "expected_original", "new_value"} - item.keys()
        if missing:
            errors.append(f"{where}: missing fields {sorted(missing)}")
            continue
        name = item["parameter"]
        if name not in constraints.allowed_parameters:
            errors.append(f"{where}: parameter {name!r} is not on the allowlist {list(constraints.allowed_parameters)}")
            continue
        target = policy.params[name]
        index = item["index"]
        if (
            not isinstance(index, list)
            or len(index) != target.ndim
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in index)
        ):
            errors.append(f"{where}: index must be a list of {target.ndim} integers")
            continue
        if any(v < 0 or v >= dim for v, dim in zip(index, target.shape)):
            errors.append(f"{where}: index {index} is outside shape {list(target.shape)}")
            continue
        key = (name, tuple(index))
        if key in seen:
            errors.append(f"{where}: duplicate target {name}{index}")
            continue
        seen.add(key)
        original, new_value = item["expected_original"], item["new_value"]
        if not (_is_number(original) and _is_number(new_value)):
            errors.append(f"{where}: expected_original and new_value must be numbers")
            continue
        if not (math.isfinite(original) and math.isfinite(new_value)):
            errors.append(f"{where}: values must be finite")
            continue
        actual = float(target[tuple(index)])
        if abs(actual - original) > constraints.original_tolerance:
            errors.append(
                f"{where}: expected_original {original} does not match baseline {actual} "
                f"(tolerance {constraints.original_tolerance})"
            )
            continue
        edit = Edit(name, tuple(index), actual, float(new_value))
        if abs(edit.delta) > constraints.max_abs_delta:
            errors.append(f"{where}: |delta| {abs(edit.delta):.3f} exceeds {constraints.max_abs_delta}")
            continue
        edits.append(edit)

    return ValidationResult(not errors, errors, edits if not errors else [], diagnosis)


def apply_edits(policy: MLPPolicy, edits: list[Edit]) -> MLPPolicy:
    """Return an edited copy. The input policy is never modified."""
    edited = policy.copy()
    for edit in edits:
        edited.params[edit.parameter][edit.index] = edit.new_value
    return edited


def weight_diff(before: MLPPolicy, after: MLPPolicy) -> list[dict]:
    """Every parameter entry that differs, recomputed from the two models."""
    changes = []
    for name, old in before.params.items():
        new = after.params[name]
        for index in zip(*np.nonzero(old != new)):
            index = tuple(int(v) for v in index)
            label = Edit(name, index, 0.0, 0.0).label()
            changes.append(
                {
                    "parameter": name,
                    "index": list(index),
                    "label": label,
                    "before": float(old[index]),
                    "after": float(new[index]),
                    "delta": round(float(new[index] - old[index]), 6),
                }
            )
    return changes
