"""Repair agent interface and the default deterministic repair agent.

A repair agent receives a ``RepairRequest`` (failing trace, feature and action
definitions, the editable policy slice, patch constraints) and returns a raw
proposal dict in the patch schema. It never sees verification fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .patch import PatchConstraints
from .policy import ACTIONS, FEATURES, HIDDEN_UNITS, MLPPolicy, output_layer_slice

FEATURE_DEFINITIONS = {
    "no_status_yet": "1 if the agent has not observed any job status yet",
    "status_running": "1 if the last observed status was 'running'",
    "status_ready": "1 if the last observed status was 'ready'",
    "is_batch": "1 if the requested job runs on the batch queue",
    "has_result": "1 if the agent already holds the fetched result",
    "last_fetch_not_ready": "1 if the previous fetch_result answered 'not_ready'",
}

ACTION_DEFINITIONS = {
    "check_status": {
        "description": "Ask the job service for the job status (running + progress, or ready).",
        "precondition_feature": None,
    },
    "fetch_result": {
        "description": "Retrieve the job output. Calling it before the job is ready is a premature fetch.",
        "precondition_feature": "status_ready",
    },
    "return_result": {
        "description": "Finish the task and return the held result to the user.",
        "precondition_feature": "has_result",
    },
}

HIDDEN_UNIT_DEFINITIONS = {
    "start": "nothing observed yet",
    "waiting": "job still running, or the last fetch was premature",
    "ready_interactive": "ready on the interactive queue, no result yet",
    "ready_batch": "ready on the batch queue, no result yet",
    "holding_result": "result in hand",
    "status_seen": "some status observed and no result yet",
    "bias": "constant unit",
}

TRACE_FIELDS = ("step", "tool", "args", "result_summary", "features", "hidden", "scores")


@dataclass(frozen=True)
class RepairRequest:
    failing_trace: list[dict]
    allowed_slice: dict
    constraints: PatchConstraints

    def to_dict(self) -> dict:
        return {
            "failing_trace": self.failing_trace,
            "features": FEATURE_DEFINITIONS,
            "actions": ACTION_DEFINITIONS,
            "hidden_units": HIDDEN_UNIT_DEFINITIONS,
            "policy": "scores = W2 @ hidden + b2; hidden = relu(W1 @ features + b1); action = argmax(scores)",
            "allowed_slice": self.allowed_slice,
            "constraints": self.constraints.to_dict(),
        }


def build_repair_request(trace: list[dict], policy: MLPPolicy, constraints: PatchConstraints) -> RepairRequest:
    clean = [{key: step[key] for key in TRACE_FIELDS} for step in trace]
    return RepairRequest(failing_trace=clean, allowed_slice=output_layer_slice(policy), constraints=constraints)


class RepairAgent(Protocol):
    label: str

    def propose(self, request: RepairRequest) -> dict: ...


class RepairUnavailable(Exception):
    """The repair agent could not produce a proposal (e.g. LLM service down, unparseable output)."""


def _repeat_runs(trace: list[dict]) -> list[list[dict]]:
    runs: list[list[dict]] = []
    for step in trace:
        key = (step["tool"], step["result_summary"])
        if runs and (runs[-1][-1]["tool"], runs[-1][-1]["result_summary"]) == key:
            runs[-1].append(step)
        else:
            runs.append([step])
    return runs


class DeterministicRepairAgent:
    """Rule-based repair computed from the trace and the allowed W2 slice.

    1. Find the longest run of identical (tool, observation) steps.
    2. The stuck state is the second step of that run: the agent has already
       seen the observation and still repeats the same tool.
    3. The target action is the action whose precondition feature is satisfied
       in the stuck state (e.g. status_ready -> fetch_result).
    4. The hidden unit to edit is the one active in the stuck state and least
       active in the other states of the trace (most specific to the loop).
    5. new_value = W2[target, unit] + (score_repeated - score_target + margin) / h_unit.
    """

    label = "Deterministic repair"
    margin = 0.5

    def propose(self, request: RepairRequest) -> dict:
        trace = request.failing_trace
        run = max(_repeat_runs(trace), key=len)
        if len(run) < 2:
            raise RepairUnavailable("no repeated action found in the trace")
        stuck = run[1]
        repeated = stuck["tool"]

        targets = [
            name
            for name, spec in ACTION_DEFINITIONS.items()
            if name != repeated and spec["precondition_feature"] and stuck["features"][spec["precondition_feature"]] == 1.0
        ]
        if not targets:
            raise RepairUnavailable("no alternative action has its precondition satisfied in the stuck state")
        target = targets[0]

        stuck_steps = {step["step"] for step in run[1:]}
        others = [step for step in trace if step["step"] not in stuck_steps]
        best_unit, best_specificity = None, 0.0
        for j, activation in enumerate(stuck["hidden"]):
            if activation <= 1e-6:
                continue
            elsewhere = max((step["hidden"][j] for step in others), default=0.0)
            specificity = activation - elsewhere
            if specificity > best_specificity + 1e-9:
                best_unit, best_specificity = j, specificity
        if best_unit is None:
            raise RepairUnavailable("no hidden unit is specific to the stuck state")

        row, col = ACTIONS.index(target), best_unit
        w2 = request.allowed_slice["values"]
        original = float(w2[row][col])
        s_rep, s_tgt = stuck["scores"][repeated], stuck["scores"][target]
        h = stuck["hidden"][col]
        new_value = round(original + (s_rep - s_tgt + self.margin) / h, 3)

        ready_feats = [name for name, v in stuck["features"].items() if v == 1.0]
        diagnosis = (
            f"Steps {run[0]['step']}-{run[-1]['step']} repeat '{repeated}' with the same observation "
            f"({stuck['result_summary']}). From step {stuck['step']} the active features are {ready_feats}, "
            f"which satisfy the precondition of '{target}', yet '{repeated}' scores {s_rep:.3f} vs "
            f"'{target}' {s_tgt:.3f}. Hidden unit '{HIDDEN_UNITS[col]}' (activation {h:.3f}) is the unit "
            f"most specific to this state (specificity {best_specificity:.3f}); its weight into '{target}' "
            f"is {original:.3f}. Raising it to {new_value:.3f} makes '{target}' win by about {self.margin} "
            f"in this state only."
        )
        return {
            "diagnosis": diagnosis,
            "edits": [{"parameter": "W2", "index": [row, col], "expected_original": original, "new_value": new_value}],
        }


# ---------------------------------------------------------------------------
# Negative controls: hand-crafted bad candidates used to show the verifier
# rejecting patches. They read only the baseline W2 values they modify.
# ---------------------------------------------------------------------------


def _w2_edit(policy: MLPPolicy, action: str, unit: str, new_value: float) -> dict:
    row, col = ACTIONS.index(action), HIDDEN_UNITS.index(unit)
    return {
        "parameter": "W2",
        "index": [row, col],
        "expected_original": float(policy.params["W2"][row, col]),
        "new_value": new_value,
    }


NEGATIVE_CONTROLS = {
    "force-return": (
        "Ends the loop by making return_result win in the batch-ready state, before anything was fetched.",
        lambda p: [_w2_edit(p, "return_result", "ready_batch", 2.0)],
    ),
    "force-fetch": (
        "Raises fetch_result on the always-on bias unit, so the agent fetches before the job is ready.",
        lambda p: [_w2_edit(p, "fetch_result", "bias", 2.9)],
    ),
    "rebalance": (
        "Moves fetch preference from the interactive-ready unit to the batch-ready unit: "
        "fixes batch jobs but breaks interactive jobs.",
        lambda p: [
            _w2_edit(p, "fetch_result", "ready_batch", 1.9),
            _w2_edit(p, "fetch_result", "ready_interactive", 0.2),
        ],
    ),
    "excessive": (
        "Exceeds the patch limits: a 10.0 jump on one weight plus four edits in total.",
        lambda p: [
            _w2_edit(p, "fetch_result", "ready_batch", 10.0),
            _w2_edit(p, "check_status", "ready_batch", -1.0),
            _w2_edit(p, "check_status", "bias", 0.0),
            _w2_edit(p, "return_result", "bias", -1.0),
        ],
    ),
}


class NegativeControlAgent:
    """Returns a fixed bad candidate. Used only to demonstrate rejection."""

    def __init__(self, name: str, policy: MLPPolicy):
        if name not in NEGATIVE_CONTROLS:
            raise ValueError(f"unknown negative control {name!r}")
        self.name = name
        self.label = f"Negative control: {name}"
        self._description, self._build = NEGATIVE_CONTROLS[name]
        self._policy = policy

    def propose(self, request: RepairRequest) -> dict:
        return {"diagnosis": f"[Hand-crafted negative control] {self._description}", "edits": self._build(self._policy)}
