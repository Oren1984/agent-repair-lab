"""Independent verification, accepted vs rejected candidates, headless path."""

import json

import pytest

from agent_repair_lab import CREDIT, cli
from agent_repair_lab.environment import load_diagnosis_case, load_verification_cases
from agent_repair_lab.experiment import run_experiment
from agent_repair_lab.patch import apply_edits, validate_proposal
from agent_repair_lab.policy import load_baseline_policy
from agent_repair_lab.repair_agent import DeterministicRepairAgent, NegativeControlAgent, build_repair_request
from agent_repair_lab.task_agent import TaskAgent
from agent_repair_lab.verifier import Verifier
from agent_repair_lab.patch import PatchConstraints


def _candidate(agent_factory):
    base = load_baseline_policy()
    trace = TaskAgent(base).run(load_diagnosis_case()).trace
    proposal = agent_factory(base).propose(build_repair_request(trace, base, PatchConstraints()))
    validation = validate_proposal(proposal, base)
    assert validation.valid, validation.errors
    return base, apply_edits(base, validation.edits)


def test_verification_cases_cover_required_kinds():
    roles = {c.role for c in load_verification_cases()}
    assert {"held_out_failure", "regression", "delayed_ready"} <= roles
    assert load_diagnosis_case().id not in {c.id for c in load_verification_cases()}


def test_repaired_policy_is_accepted():
    base, edited = _candidate(lambda b: DeterministicRepairAgent())
    verdict = Verifier(10).verify(base, edited, load_diagnosis_case(), load_verification_cases())
    assert verdict.accepted, verdict.reasons
    by_role = {}
    for c in verdict.candidate:
        by_role.setdefault(c.role, []).append(c)
        assert c.correct and not c.loop_detected and c.premature_fetches == 0
    # held-out failure: baseline fails, candidate passes
    held = [b for b in verdict.baseline if b.role == "held_out_failure"]
    assert held and not any(b.passed for b in held)


@pytest.mark.parametrize(
    "name, expected_reason",
    [
        ("force-return", "wrong result"),
        ("force-fetch", "premature fetch"),
        ("rebalance", "REGRESSION"),
    ],
)
def test_bad_patches_that_end_the_loop_are_rejected(name, expected_reason):
    base, edited = _candidate(lambda b: NegativeControlAgent(name, b))
    diag = load_diagnosis_case()
    edited_run = TaskAgent(edited).run(diag)
    assert edited_run.termination == "returned_result"  # the loop is gone...
    verdict = Verifier(10).verify(base, edited, diag, load_verification_cases())
    assert not verdict.accepted  # ...but the verifier still rejects it
    assert any(expected_reason in r for r in verdict.reasons), verdict.reasons


def test_rebalance_passes_diagnosis_and_held_out_but_fails_regression():
    base, edited = _candidate(lambda b: NegativeControlAgent("rebalance", b))
    verdict = Verifier(10).verify(base, edited, load_diagnosis_case(), load_verification_cases())
    outcome = {c.scenario_id: c for c in verdict.candidate}
    assert outcome["diag-batch-J1042"].passed
    assert outcome["heldout-batch-J2077"].passed
    assert not outcome["regression-interactive-J3300"].passed


def test_experiment_accepts_deterministic_repair():
    report = run_experiment("deterministic")
    assert report["status"] == "accepted" and report["exit_code"] == 0
    assert report["original_weights_unchanged"]
    cmp = report["comparison"]
    assert cmp["baseline"]["termination"] == "budget_exhausted" and cmp["baseline"]["loop_detected"]
    assert cmp["guarded"]["termination"] == "repeat_guard_stopped" and not cmp["guarded"]["correct"]
    assert cmp["edited"]["termination"] == "returned_result" and cmp["edited"]["correct"]
    assert len(report["weight_changes"]) == 1
    json.dumps(report)  # fully serialisable


@pytest.mark.parametrize("source, status", [("negative:rebalance", "rejected"), ("negative:excessive", "invalid_proposal")])
def test_experiment_reports_rejections(source, status):
    report = run_experiment(source)
    assert report["status"] == status
    assert report["original_weights_unchanged"]
    if status == "invalid_proposal":
        assert report["comparison"]["edited"] is None and report["weight_changes"] == []


def test_cli_uses_same_experiment(capsys):
    code = cli.main(["--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    ref = run_experiment("deterministic")
    assert out["status"] == ref["status"]
    assert out["weight_changes"] == ref["weight_changes"]
    assert out["credit"] == CREDIT == "Built and developed by Oren Salami · V1.0"


def test_cli_exit_codes(capsys):
    assert cli.main(["--source", "negative:force-return"]) == 1
    assert cli.main(["--source", "negative:excessive"]) == 2
    assert "VERDICT: INVALID_PROPOSAL" in capsys.readouterr().out
