"""Tiny deterministic NumPy MLP that chooses the task agent's next tool call.

Architecture: 6 state features -> 7 ReLU hidden units -> 3 action scores.
The action with the highest score is executed (argmax, ties -> lowest index).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources

import numpy as np

FEATURES = (
    "no_status_yet",  # 1 if the agent has not observed any job status yet
    "status_running",  # 1 if the last observed status was "running"
    "status_ready",  # 1 if the last observed status was "ready"
    "is_batch",  # 1 if the requested job runs on the batch queue
    "has_result",  # 1 if the agent already holds the fetched result
    "last_fetch_not_ready",  # 1 if the previous fetch_result answered "not_ready"
)

HIDDEN_UNITS = (
    "start",  # h0: nothing observed yet
    "waiting",  # h1: job still running / fetch was premature
    "ready_interactive",  # h2: ready, interactive queue, no result yet
    "ready_batch",  # h3: ready, batch queue, no result yet
    "holding_result",  # h4: result in hand
    "status_seen",  # h5: some status observed and no result yet
    "bias",  # h6: constant unit
)

ACTIONS = ("check_status", "fetch_result", "return_result")

PARAMETER_NAMES = ("W1", "b1", "W2", "b2")


@dataclass(frozen=True)
class Forward:
    hidden: np.ndarray
    scores: np.ndarray

    @property
    def action_index(self) -> int:
        return int(np.argmax(self.scores))

    @property
    def action(self) -> str:
        return ACTIONS[self.action_index]


class MLPPolicy:
    """Holds the numeric parameters. Instances are never mutated by the lab."""

    def __init__(self, params: dict[str, np.ndarray]):
        self.params = {name: np.array(params[name], dtype=float) for name in PARAMETER_NAMES}
        self._check_shapes()

    def _check_shapes(self) -> None:
        n_in, n_hidden, n_out = len(FEATURES), len(HIDDEN_UNITS), len(ACTIONS)
        expected = {"W1": (n_hidden, n_in), "b1": (n_hidden,), "W2": (n_out, n_hidden), "b2": (n_out,)}
        for name, shape in expected.items():
            if self.params[name].shape != shape:
                raise ValueError(f"{name} has shape {self.params[name].shape}, expected {shape}")

    def forward(self, features: np.ndarray) -> Forward:
        p = self.params
        hidden = np.maximum(0.0, p["W1"] @ features + p["b1"])
        scores = p["W2"] @ hidden + p["b2"]
        return Forward(hidden=hidden, scores=scores)

    def copy(self) -> "MLPPolicy":
        return MLPPolicy({name: value.copy() for name, value in self.params.items()})

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name in PARAMETER_NAMES:
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(self.params[name]).tobytes())
        return digest.hexdigest()[:16]


def load_baseline_policy() -> MLPPolicy:
    text = resources.files("agent_repair_lab.fixtures").joinpath("baseline_policy.json").read_text()
    raw = json.loads(text)
    return MLPPolicy({name: np.array(raw[name], dtype=float) for name in PARAMETER_NAMES})


def output_layer_slice(policy: MLPPolicy) -> dict:
    """The only part of the policy a repair agent may inspect and edit."""
    return {
        "parameter": "W2",
        "shape": list(policy.params["W2"].shape),
        "rows": list(ACTIONS),
        "columns": list(HIDDEN_UNITS),
        "values": policy.params["W2"].round(6).tolist(),
    }
