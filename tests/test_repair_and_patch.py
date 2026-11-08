"""Deterministic repair, patch validation and application."""

import copy
import math

import numpy as np
import pytest

from agent_repair_lab.environment import load_diagnosis_case
from agent_repair_lab.patch import PatchConstraints, apply_edits, validate_proposal, weight_diff
from agent_repair_lab.policy import ACTIONS, HIDDEN_UNITS, load_baseline_policy
from agent_repair_lab.repair_agent import (
    DeterministicRepairAgent,
    NegativeControlAgent,
    RepairUnavailable,
    build_repair_request,
)
from agent_repair_lab.task_agent import TaskAgent

C = PatchConstraints()


def _request(trace=None):
    base = load_baseline_policy()
    trace = trace or TaskAgent(base).run(load_diagnosis_case()).trace
    return build_repair_request(trace, base, C)


def test_deterministic_repair_derives_edit_from_trace():
    proposal = DeterministicRepairAgent().propose(_request())
    assert len(proposal["edits"]) == 1
    edit = proposal["edits"][0]
    assert edit["parameter"] == "W2"
    assert edit["index"] == [ACTIONS.index("fetch_result"), HIDDEN_UNITS.index("ready_batch")]
    base = load_baseline_policy()
    assert edit["expected_original"] == pytest.approx(base.params["W2"][1, 3])
    assert "check_status" in proposal["diagnosis"] and "fetch_result" in proposal["diagnosis"]
    assert validate_proposal(proposal, base, C).valid


def test_deterministic_repair_responds_to_its_input_scores():
    """Change the observed scores in the trace and the computed edit changes: not a canned patch."""
    request = _request()
    altered = copy.deepcopy(request.failing_trace)
    for step in altered:
        step["scores"]["check_status"] += 0.4
    a = DeterministicRepairAgent().propose(request)["edits"][0]["new_value"]
    b = DeterministicRepairAgent().propose(build_repair_request(altered, load_baseline_policy(), C))["edits"][0]["new_value"]
    assert b == pytest.approx(a + 0.4, abs=2e-3)


def test_deterministic_repair_refuses_trace_without_loop():
    base = load_baseline_policy()
    from agent_repair_lab.environment import load_verification_cases

    good = next(c for c in load_verification_cases() if c.id == "regression-interactive-J3300")
    trace = TaskAgent(base).run(good).trace
    with pytest.raises(RepairUnavailable):
        DeterministicRepairAgent().propose(build_repair_request(trace, base, C))


def test_repair_request_contains_no_verification_fixtures():
    text = str(_request().to_dict())
    for forbidden in ("J-2077", "J-3300", "J-3815", "J-4105", "expected_result", "held_out"):
        assert forbidden not in text


def test_apply_changes_only_intended_weight_and_preserves_original():
    base = load_baseline_policy()
    before = {k: v.copy() for k, v in base.params.items()}
    validation = validate_proposal(DeterministicRepairAgent().propose(_request()), base, C)
    edited = apply_edits(base, validation.edits)
    diff = weight_diff(base, edited)
    assert [(d["parameter"], d["index"]) for d in diff] == [("W2", [1, 3])]
    for name in before:
        np.testing.assert_array_equal(base.params[name], before[name])
    assert edited.params["W2"][1, 3] == validation.edits[0].new_value


def _edit(**overrides):
    base = load_baseline_policy()
    e = {"parameter": "W2", "index": [1, 3], "expected_original": float(base.params["W2"][1, 3]), "new_value": 1.4}
    e.update(overrides)
    return e


@pytest.mark.parametrize(
    "proposal, fragment",
    [
        ("not a dict", "JSON object"),
        ({"diagnosis": "x"}, "edits must be a list"),
        ({"diagnosis": "x", "edits": []}, "no edits"),
        ({"diagnosis": "", "edits": [_edit()]}, "diagnosis"),
        ({"diagnosis": "x", "edits": [_edit(parameter="W1", index=[0, 0])]}, "allowlist"),
        ({"diagnosis": "x", "edits": [_edit(parameter="b2", index=[1])]}, "allowlist"),
        ({"diagnosis": "x", "edits": [_edit(index=[3, 0])]}, "outside shape"),
        ({"diagnosis": "x", "edits": [_edit(index=[1])]}, "index must be"),
        ({"diagnosis": "x", "edits": [_edit(index=[1.0, 3])]}, "index must be"),
        ({"diagnosis": "x", "edits": [_edit(), _edit()]}, "duplicate"),
        ({"diagnosis": "x", "edits": [_edit(new_value=math.nan)]}, "finite"),
        ({"diagnosis": "x", "edits": [_edit(new_value=math.inf)]}, "finite"),
        ({"diagnosis": "x", "edits": [_edit(new_value="1.4")]}, "numbers"),
        ({"diagnosis": "x", "edits": [_edit(new_value=9.0)]}, "exceeds"),
        ({"diagnosis": "x", "edits": [_edit(expected_original=0.5)]}, "does not match"),
        ({"diagnosis": "x", "edits": [{"parameter": "W2"}]}, "missing fields"),
        (
            {"diagnosis": "x", "edits": [_edit(index=[0, i]) for i in range(4)]},
            "exceed the limit",
        ),
    ],
)
def test_invalid_proposals_rejected_before_any_mutation(proposal, fragment):
    base = load_baseline_policy()
    fp = base.fingerprint()
    result = validate_proposal(proposal, base, C)
    assert not result.valid
    assert result.edits == []
    assert any(fragment in e for e in result.errors), result.errors
    assert base.fingerprint() == fp


def test_one_bad_edit_invalidates_whole_proposal():
    base = load_baseline_policy()
    result = validate_proposal({"diagnosis": "x", "edits": [_edit(), _edit(index=[0, 0], new_value=50.0)]}, base, C)
    assert not result.valid and result.edits == []


def test_excessive_negative_control_is_invalid():
    base = load_baseline_policy()
    proposal = NegativeControlAgent("excessive", base).propose(_request())
    result = validate_proposal(proposal, base, C)
    assert not result.valid
    assert any("exceed the limit" in e for e in result.errors)
    assert any("|delta|" in e for e in result.errors)
